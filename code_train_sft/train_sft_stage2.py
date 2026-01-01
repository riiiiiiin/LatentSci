import torch
import torch.nn as nn
from trl import SFTTrainer, SFTConfig
from transformers import AutoTokenizer, TrainerCallback, DataCollatorForSeq2Seq
from dataloader import load_data
from model_new import Qwen3MoleculeLLM
import os
import time
import json
import logging
import wandb
from config import ModelConfig
from typing import Dict, List, Any, Optional
from peft import LoraConfig, get_peft_model, TaskType, PeftModel, PeftConfig


# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 自定义回调函数，用于监控训练过程
class LoraTrainingMonitorCallback(TrainerCallback):
    """LoRA训练监控回调函数：仅负责打印关键信息，不重复记录 wandb"""
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """同时打印日志到控制台"""
        if logs:
            if 'loss' in logs:
                logger.info(f"Step {state.global_step}: loss = {logs['loss']:.4f}")
            if 'learning_rate' in logs:
                logger.info(f"Step {state.global_step}: lr = {logs['learning_rate']:.6f}")
    
    def on_train_begin(self, args, state, control, **kwargs):
        """训练开始时记录LoRA参数信息"""
        if 'model' in kwargs:
            model = kwargs['model']
            # 打印可训练参数信息
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            logger.info(f"LoRA模型参数统计:")
            logger.info(f"  总参数: {total_params:,}")
            logger.info(f"  可训练参数: {trainable_params:,}")
            logger.info(f"  可训练比例: {100 * trainable_params / total_params:.2f}%")
            
            # 记录到wandb
            if wandb.run is not None:
                wandb.config.update({
                    "total_params": total_params,
                    "trainable_params": trainable_params,
                    "trainable_ratio": 100 * trainable_params / total_params
                })
            
            # 打印LoRA适配器信息
            if hasattr(model, 'peft_config'):
                for adapter_name, config in model.peft_config.items():
                    logger.info(f"  LoRA配置 - {adapter_name}:")
                    logger.info(f"    r={config.r}, alpha={config.lora_alpha}, dropout={config.lora_dropout}")
                    
                    # 记录到wandb
                    if wandb.run is not None:
                        wandb.config.update({
                            f"lora_{adapter_name}_r": config.r,
                            f"lora_{adapter_name}_alpha": config.lora_alpha,
                            f"lora_{adapter_name}_dropout": config.lora_dropout
                        })
    
    def on_save(self, args, state, control, **kwargs):
        """保存检查点时记录"""
        if wandb.run is not None:
            wandb.log({"checkpoint_step": state.global_step})
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """每个epoch结束时记录"""
        if wandb.run is not None:
            wandb.log({"epoch": state.epoch})

# 工业级多模态数据整理器
class MultiModalDataCollator(DataCollatorForSeq2Seq):
    """
    继承自 DataCollatorForSeq2Seq，支持自动文本补齐和 Label 掩码，
    同时兼容自定义的 smiles 字段。
    """
    def __call__(self, features):
        # 1. 提取并移除不支持的 smiles 字段
        smiles = [f.pop("smiles") for f in features]
        
        # 2. 调用父类的 __call__ 进行标准补齐
        # 它会自动将 input_ids 补齐到 pad_token_id，将 labels 补齐到 -100
        batch = super().__call__(features)
        
        # 3. 放回 smiles 字段
        batch["smiles"] = smiles
        return batch

# 加载已训练的模型组件
def load_trained_components(
    model,
    lora_weights_path=None,
    projector_path=None,
    device=None
):
    """
    加载已训练的组件到模型。
    取消所有 fallback 策略，加载失败即报错，确保训练基点正确。
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 加载 LoRA 权重
    if lora_weights_path:
        if not os.path.exists(lora_weights_path):
            raise FileNotFoundError(f"LoRA weights path not found: {lora_weights_path}")
            
        logger.info(f"Loading LoRA weights from: {lora_weights_path}")
        # 强制使用 PeftModel 加载，不进行任何 try-except 容错
        model.model = PeftModel.from_pretrained(
            model.model, 
            lora_weights_path,
            is_trainable=True 
        )
        logger.info("Successfully loaded LoRA weights.")
    
    # 2. 加载投影器权重
    if projector_path:
        if not os.path.exists(projector_path):
            raise FileNotFoundError(f"Projector weights path not found: {projector_path}")
            
        logger.info(f"Loading projector weights from: {projector_path}")
        # 强制加载状态字典
        projector_state_dict = torch.load(projector_path, map_location=device)
        model.projector.load_state_dict(projector_state_dict)
        logger.info("Successfully loaded projector weights.")
    
    return model

def train_sft_lora(
    model_name=None,
    data_path=None,
    output_dir=ModelConfig.DEFAULT_OUTPUT_DIR,
    epochs=3,
    batch_size=32,
    lr=2e-4,
    max_seq_length=8192,
    resume_from_checkpoint=None,  # 新增：从检查点恢复训练
    lora_weights_path=None,  # 新增：预训练的LoRA权重路径
    projector_path=None,  # 新增：预训练的投影器权重路径
    wandb_project="qwen3-molecule-lora-sft",
    wandb_run_name=None,
    wandb_entity=None,
    mol_config=None,  # 新增：显式传入分子配置
    include_cot=True, # 新增：是否包含 CoT
):
    """
    使用LoRA进行SFT训练，支持从预训练权重继续训练
    """
    # 移除 os.environ["WANDB_MODE"] = "disabled"，由 SFTConfig 控制
    
    if model_name is None:
        model_name = ModelConfig.DEFAULT_QWEN_PATH
    if data_path is None:
        data_path = ModelConfig.DEFAULT_DATA_PATH
    
    # 如果没传 mol_config，则使用默认配置
    if mol_config is None:
        mol_config = {
            'num_queries': ModelConfig.NUM_QUERIES,
            'input_dim': ModelConfig.INPUT_DIM,
            'num_heads': ModelConfig.NUM_HEADS
        }
    
    logger.info("=" * 60)
    logger.info("LoRA SFT Training for Qwen3MoleculeLLM")
    logger.info(f"Resume from: {resume_from_checkpoint or 'scratch'}")
    logger.info("=" * 60)
    
    # ============================
    # 0. 初始化wandb
    # ============================
    logger.info(f"Initializing wandb for experiment tracking...")
    # 如果未指定run_name，自动生成一个包含时间戳的名称
    if wandb_run_name is None:
        from datetime import datetime
        wandb_run_name = f"qwen3-lora-sft-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    # 初始化wandb (取消 try-except，如果失败建议检查网络或设置 WANDB_MODE=offline)
    wandb.init(
        project=wandb_project,
        name=wandb_run_name,
        entity=wandb_entity,
        config={
            "model_name": model_name,
            "data_path": data_path,
            "output_dir": output_dir,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "max_seq_length": max_seq_length,
            "training_strategy": "LoRA-SFT",
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "optimizer": "adamw_8bit",
            "mixed_precision": "bf16",
            "gradient_accumulation": 4,
            "resume_from_checkpoint": resume_from_checkpoint,
            "lora_weights_path": lora_weights_path,
            "projector_path": projector_path,
            "num_queries": mol_config['num_queries'],
            "include_cot": include_cot,
            "training_stage": "second_stage" if (resume_from_checkpoint or lora_weights_path) else "first_stage"
        }
    )
    logger.info(f"Wandb initialized: project={wandb_project}, run={wandb_run_name}")
    
    # ============================
    # 1. 初始化基础模型
    # ============================
    logger.info("Initializing base model...")
    # 传入 mol_config
    model = Qwen3MoleculeLLM(qwen_model_name=model_name, mol_config=mol_config, device_map=None)
    tokenizer = model.tokenizer
    
    # 确保pad_token设置
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Set pad_token to eos_token: {tokenizer.pad_token}")
    
    # ============================
    # 2. 加载预训练权重（如果提供）
    # ============================
    if lora_weights_path or projector_path:
        logger.info("Loading pre-trained components...")
        model = load_trained_components(
            model,
            lora_weights_path=lora_weights_path,
            projector_path=projector_path
        )
    
    # ============================
    # 3. 配置LoRA（如果还没有LoRA）
    # ============================
    if not hasattr(model.model, 'peft_config') or model.model.peft_config is None:
        logger.info("Configuring LoRA from scratch...")
        
        # 首先冻结所有参数
        logger.info("Freezing base model parameters...")
        for param in model.parameters():
            param.requires_grad = False
        
        # LoRA配置
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,  # LoRA秩
            lora_alpha=32,  # LoRA alpha
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
        )
        
        # 将LoRA适配器添加到LLM部分
        logger.info("Adding LoRA adapters to LLM...")
        model.model = get_peft_model(model.model, lora_config)
    else:
        logger.info("Using existing LoRA configuration")
        # 确保LoRA参数可训练
        for param in model.model.parameters():
            if hasattr(param, 'requires_grad'):
                param.requires_grad = True
    
    # 确保投影器可训练
    logger.info("Keeping projector trainable...")
    for param in model.projector.parameters():
        param.requires_grad = True
    
    # ============================
    # 4. 加载数据集
    # ============================
    logger.info(f"Loading dataset from {data_path} (Include CoT: {include_cot}, Max Len: {max_seq_length})...")
    train_dataset = load_data(data_path, include_cot=include_cot, max_len=max_seq_length)
    logger.info(f"Dataset loaded: {len(train_dataset)} samples")
    
    # 记录数据集信息到wandb
    if wandb.run is not None:
        wandb.config.update({
            "dataset_size": len(train_dataset),
            "dataset_path": data_path,
        })
    
    # ============================
    # 5. 配置训练参数
    # ============================
    logger.info("Configuring training arguments...")
    
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=lr,
        bf16=True,
        max_seq_length=max_seq_length,
        packing=False,
        dataset_text_field=None,
        remove_unused_columns=False,
        logging_steps=10,
        eval_strategy="no", # 不做划分，直接全部训练
        save_steps=100,
        save_total_limit=3,
        gradient_checkpointing=True, # 🚨 针对 8192 长度默认开启，防止 OOM
        max_grad_norm=0.3,
        warmup_ratio=0.1,
        report_to="wandb", # 🚨 开启官方 wandb 支持
        dataloader_pin_memory=False,
        dataloader_num_workers=2,
        optim="adamw_8bit",
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir, "logs"),
        resume_from_checkpoint=resume_from_checkpoint,
    )
    
    # 打印训练配置
    logger.info(f"Training Configuration:")
    logger.info(f"  Model: {model_name}")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Learning rate: {lr}")
    logger.info(f"  Max sequence length: {max_seq_length}")
    logger.info(f"  Resume from: {resume_from_checkpoint or 'None'}")
    logger.info(f"  Pre-trained LoRA: {lora_weights_path or 'None'}")
    logger.info(f"  Pre-trained projector: {projector_path or 'None'}")
    
    # ============================
    # 6. 初始化SFTTrainer
    # ============================
    logger.info("Initializing SFTTrainer...")
    
    data_collator = MultiModalDataCollator(
        tokenizer=tokenizer,
        model=model.model,
        padding=True,
    )
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[LoraTrainingMonitorCallback()],
    )
    
    # ============================
    # 7. 训练前验证
    # ============================
    logger.info("Testing forward pass...")
    try:
        if len(train_dataset) > 0:
            test_sample = [train_dataset[0]]
            test_batch = data_collator(test_sample)
            
            # 获取模型主显卡
            device = next(model.parameters()).device
            
            test_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in test_batch.items()}
            
            with torch.no_grad():
                outputs = model(
                    input_ids=test_batch["input_ids"],
                    attention_mask=test_batch["attention_mask"],
                    labels=test_batch["labels"],
                    smiles=test_batch["smiles"]
                )

            logger.info(f"Forward test successful!")
            initial_loss = outputs.loss.item() if outputs.loss is not None else float('nan')
            logger.info(f"  Initial Loss: {initial_loss}")
            
            if wandb.run is not None:
                wandb.log({"initial_loss": initial_loss})
            
            del test_sample, test_batch, outputs
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
    except Exception as e:
        logger.error(f"Forward test failed: {e}")
        raise
    
    # ============================
    # 8. 训练
    # ============================
    logger.info("Starting LoRA training...")
    
    try:
        start_time = time.time()
        
        # 开始训练
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        training_time = time.time() - start_time
        hours, remainder = divmod(training_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        logger.info("LoRA training completed!")
        logger.info(f"Total training time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
        
        if wandb.run is not None:
            wandb.log({"total_training_time_hours": training_time / 3600})
            wandb.config.update({"total_training_time_seconds": training_time})
        
        # ============================
        # 9. 保存模型
        # ============================
        logger.info("Saving models...")
        
        # 保存完整的模型
        trainer.save_model(output_dir)
        
        # 单独保存LoRA适配器权重
        lora_weights_path_save = os.path.join(output_dir, "lora_weights")
        os.makedirs(lora_weights_path_save, exist_ok=True)
        model.model.save_pretrained(lora_weights_path_save)
        logger.info(f"LoRA weights saved to: {lora_weights_path_save}")
        
        # 保存投影器权重
        projector_path_save = os.path.join(output_dir, "projector.pt")
        torch.save(model.projector.state_dict(), projector_path_save)
        logger.info(f"Projector weights saved to: {projector_path_save}")
        
        # 保存分词器
        tokenizer.save_pretrained(output_dir)
        
        # 保存训练配置
        config_save_path = os.path.join(output_dir, "training_config.json")
        config_dict = {
            "model_name": model_name,
            "data_path": data_path,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "max_seq_length": max_seq_length,
            "resume_from_checkpoint": resume_from_checkpoint,
            "pretrained_lora": lora_weights_path,
            "pretrained_projector": projector_path,
            "training_time_seconds": training_time,
            "training_stage": "second_stage" if (resume_from_checkpoint or lora_weights_path) else "first_stage"
        }
        with open(config_save_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        if wandb.run is not None:
            artifact = wandb.Artifact(
                name=f"training-config-{wandb_run_name}",
                type="config",
                description="Training configuration file"
            )
            artifact.add_file(config_save_path)
            wandb.log_artifact(artifact)
        
        # ============================
        # 10. 合并LoRA权重（可选）
        # ============================
        logger.info("Merging LoRA weights with base model...")
        try:
            merged_model = model.model.merge_and_unload()
            
            merged_model_path = os.path.join(output_dir, "merged_model")
            os.makedirs(merged_model_path, exist_ok=True)
            
            merged_model.save_pretrained(merged_model_path)
            tokenizer.save_pretrained(merged_model_path)
            
            torch.save(model.projector.state_dict(), os.path.join(merged_model_path, "projector.pt"))
            
            logger.info(f"Merged model saved to: {merged_model_path}")
            
            if wandb.run is not None:
                lora_artifact = wandb.Artifact(
                    name=f"lora-weights-{wandb_run_name}",
                    type="model",
                    description="LoRA adapter weights"
                )
                lora_artifact.add_dir(lora_weights_path_save)
                wandb.log_artifact(lora_artifact)
                
                model_artifact = wandb.Artifact(
                    name=f"full-model-{wandb_run_name}",
                    type="model",
                    description="Full model with merged LoRA weights"
                )
                model_artifact.add_dir(merged_model_path)
                wandb.log_artifact(model_artifact)
            
        except Exception as e:
            logger.warning(f"Failed to merge LoRA weights: {e}")
            logger.warning("Using unmerged model for inference")

        # 完成wandb运行
        if wandb.run is not None:
            wandb.finish()
        
        return model
        
    except torch.cuda.OutOfMemoryError:
        logger.error("CUDA out of memory during LoRA training!")
        logger.error("Try reducing batch_size or max_seq_length")
        if wandb.run is not None:
            # 记录 OOM 事件到 wandb
            wandb.log({"error/oom": 1})
            wandb.finish(exit_code=1)
        raise
    
    except Exception as e:
        logger.error(f"LoRA training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        if wandb.run is not None:
            wandb.finish(exit_code=1)
        raise

# 加载LoRA模型进行推理
def load_lora_model_for_inference(
    base_model_path=None,
    lora_weights_path="./qwen3_mol_sft_lora_results/lora_weights",
    projector_path="./qwen3_mol_sft_lora_results/projector.pt",
    merge_lora=True,
    device="cuda" if torch.cuda.is_available() else "cpu",
    mol_config=None  # 新增：分子配置
):
    """
    加载LoRA微调的模型进行推理 - 修复设备一致性
    """
    if base_model_path is None:
        base_model_path = ModelConfig.DEFAULT_QWEN_PATH
    
    if mol_config is None:
        mol_config = {
            'num_queries': ModelConfig.NUM_QUERIES,
            'input_dim': ModelConfig.INPUT_DIM,
            'num_heads': ModelConfig.NUM_HEADS
        }

    logger.info(f"Loading LoRA model for inference on {device}...")
    
    # 1. 加载基础模型，传入 mol_config
    model = Qwen3MoleculeLLM(qwen_model_name=base_model_path, mol_config=mol_config)
    tokenizer = model.tokenizer
    
    # 2. 确保pad_token设置
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 3. 将模型移动到指定设备
    model = model.to(device)
    
    # 4. 加载LoRA权重
    from peft import PeftModel
    model.model = PeftModel.from_pretrained(model.model, lora_weights_path)
    
    # 5. 合并LoRA权重（可选，用于更快推理）
    if merge_lora:
        logger.info("Merging LoRA weights for faster inference...")
        model.model = model.model.merge_and_unload()
    
    # 6. 加载投影器权重并确保在正确设备上
    if os.path.exists(projector_path):
        # 加载时指定map_location
        projector_state_dict = torch.load(projector_path, map_location=device)
        model.projector.load_state_dict(projector_state_dict)
        # 确保投影器在正确设备上
        model.projector = model.projector.to(device)
        logger.info(f"Loaded projector weights to {device} from: {projector_path}")
    
    # 7. 确保模型各部分都在同一设备上
    model = model.to(device)
    
    # 8. 检查设备一致性
    model_devices = set()
    for name, param in model.named_parameters():
        model_devices.add(str(param.device))
    
    if len(model_devices) > 1:
        logger.warning(f"Model parameters are on multiple devices: {model_devices}")
        # 强制统一设备
        model = model.to(device)
    
    logger.info(f"Model loaded successfully on {device}")
    
    # 9. 设置为评估模式
    model.eval()
    
    return model, tokenizer


# 加载测试数据
def load_test_data(test_data_path):
    """
    从JSON文件加载测试数据
    
    Args:
        test_data_path: 测试数据路径（可以是文件或目录）
    
    Returns:
        test_cases: 测试用例列表，每个元素包含 {smiles, query, label (if available), cot (if available)}
    """
    import glob
    from datasets import load_dataset
    
    logger.info(f"Loading test data from: {test_data_path}")
    
    # 判断是文件还是目录
    if os.path.isfile(test_data_path):
        data_files = [test_data_path]
    elif os.path.isdir(test_data_path):
        # 扫描所有 JSON 文件并排除 rxn/rcr.json
        all_json_files = glob.glob(os.path.join(test_data_path, "**/*.json"), recursive=True)
        data_files = [f for f in all_json_files if not f.endswith("rcr.json")]
    else:
        raise ValueError(f"Test data path not found: {test_data_path}")
    
    logger.info(f"Found {len(data_files)} test files:")
    for f in sorted(data_files):
        logger.info(f"  - {f}")
    
    # 加载数据
    ds = load_dataset("json", data_files=data_files)["train"]
    
    # 提取字段（使用 dataloader 中的 extract_fields 函数）
    from dataloader import extract_fields
    test_cases = []
    
    for example in ds:
        try:
            processed = extract_fields(example)
            test_cases.append({
                "smiles": processed["input_smiles"],
                "query": processed["query"],
                "label": processed.get("label", None),
                "cot": processed.get("cot", None)
            })
        except Exception as e:
            logger.warning(f"Failed to process test example: {e}")
            continue
    
    logger.info(f"Loaded {len(test_cases)} test cases")
    return test_cases


# 推理测试函数
def run_inference_on_test_data(
    model,
    tokenizer,
    test_data_path,
    max_new_tokens=2048,
    temperature=0.7,
    top_p=0.9,
    device="cuda" if torch.cuda.is_available() else "cpu",
    save_results_path=None,
    batch_size=1,
    max_samples=None
):
    """
    在测试数据上运行推理
    
    Args:
        model: 加载的模型
        tokenizer: 分词器
        test_data_path: 测试数据路径（文件或目录）
        max_new_tokens: 最大生成token数
        temperature: 温度参数
        top_p: top-p采样参数
        device: 设备
        save_results_path: 保存结果的文件路径
        batch_size: 批次大小（当前仅支持1）
        max_samples: 最大测试样本数（None表示全部测试）
    
    Returns:
        results: 推理结果列表
    """
    logger.info(f"Running inference on test data from {test_data_path}")
    logger.info(f"Device: {device}, Max new tokens: {max_new_tokens}")
    
    # 加载测试数据
    test_cases = load_test_data(test_data_path)
    
    # 限制测试样本数
    if max_samples is not None and max_samples < len(test_cases):
        logger.info(f"Limiting test to first {max_samples} samples")
        test_cases = test_cases[:max_samples]
    
    # 确保模型在正确设备上
    model.eval()
    model = model.to(device)
    
    results = []
    
    # 逐个样本推理
    for idx, test_case in enumerate(test_cases):
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing sample {idx+1}/{len(test_cases)}")
        logger.info(f"Input SMILES: {test_case['smiles']}")
        logger.info(f"Input query: {test_case['query'][:200]}...")  # 显示前200字符
        
        try:
            # 清理SMILES（移除点号分隔符）
            cleaned_smiles = [s.replace(".", "").strip() for s in test_case['smiles']]
            
            # 编码提示文本
            encodings = tokenizer(
                test_case['query'],
                padding=True,
                truncation=True,
                max_length=2048,
                return_tensors="pt"
            )
            
            # 确保所有输入在相同设备上
            input_ids = encodings["input_ids"].to(device)
            attention_mask = encodings["attention_mask"].to(device)
            
            # 生成回复
            with torch.no_grad():
                generated_ids = model.generate(
                    smiles_list=[cleaned_smiles],
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True if temperature > 0 else False,
                )
            
            # 解码结果
            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            logger.info(f"Generated response: {generated_text[:200]}...")  # 显示前200字符
            
            # 保存结果
            result = {
                "sample_id": idx,
                "smiles": test_case['smiles'],
                "query": test_case['query'],
                "generated_response": generated_text.strip(),
                "ground_truth_label": test_case.get('label', None),
                "ground_truth_cot": test_case.get('cot', None)
            }
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error processing sample {idx}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            results.append({
                "sample_id": idx,
                "smiles": test_case['smiles'],
                "query": test_case['query'],
                "generated_response": None,
                "error": str(e),
                "ground_truth_label": test_case.get('label', None),
                "ground_truth_cot": test_case.get('cot', None)
            })
        
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Inference completed on {len(results)} samples")
    
    # 保存结果
    if save_results_path:
        from datetime import datetime
        
        # 准备保存的数据
        save_data = {
            "timestamp": datetime.now().isoformat(),
            "test_data_path": test_data_path,
            "model_info": {
                "device": str(device),
                "total_parameters": sum(p.numel() for p in model.parameters()),
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            },
            "generation_config": {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
            },
            "num_samples": len(results),
            "test_results": results
        }
        
        # 确保目录存在
        os.makedirs(os.path.dirname(save_results_path) if os.path.dirname(save_results_path) else ".", exist_ok=True)
        
        # 保存到JSON文件
        with open(save_results_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to: {save_results_path}")
    
    return results


# 主函数 - 修改以支持第二轮训练
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LoRA微调多模态分子-语言模型")
    parser.add_argument("--mode", type=str, choices=["train", "inference"], default="train", help="运行模式")
    parser.add_argument("--output_dir", type=str, default=ModelConfig.DEFAULT_OUTPUT_DIR, help="保存/加载模型的路径")
    parser.add_argument("--data_path", type=str, default=ModelConfig.DEFAULT_DATA_PATH, help="数据路径")
    parser.add_argument("--batch_size", type=int, default=2, help="批次大小")
    parser.add_argument("--max_seq_length", type=int, default=8192, help="最大序列长度")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--include_cot", type=lambda x: (str(x).lower() == 'true'), default=True, help="是否在 Response 中包含思维链 (CoT)")
    
    # 权重加载参数
    parser.add_argument("--lora_path", type=str, default=None, help="预训练 LoRA 权重路径")
    parser.add_argument("--projector_path", type=str, default=None, help="预训练投影器权重路径")
    
    # 分子模型超参数
    parser.add_argument("--num_queries", type=int, default=ModelConfig.NUM_QUERIES, help="投影器查询向量数量")
    parser.add_argument("--mol_input_dim", type=int, default=ModelConfig.INPUT_DIM, help="分子编码器输出维度")
    parser.add_argument("--mol_num_heads", type=int, default=ModelConfig.NUM_HEADS, help="投影器注意力头数")
    
    parser.add_argument("--resume_checkpoint", type=str, default=None, help="从检查点恢复训练")
    parser.add_argument("--wandb_project", type=str, default="qwen3-molecule-lora-sft", help="wandb项目名称")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="wandb运行名称")
    parser.add_argument("--wandb_entity", type=str, default=None, help="wandb团队/实体名称")
    
    # 推理模式参数
    parser.add_argument("--test_data_path", type=str, default=None, help="测试数据路径（用于inference模式）")
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="生成的最大token数（用于inference模式）")
    parser.add_argument("--temperature", type=float, default=0.7, help="生成温度（用于inference模式）")
    parser.add_argument("--top_p", type=float, default=0.9, help="top-p采样参数（用于inference模式）")
    parser.add_argument("--max_test_samples", type=int, default=None, help="最大测试样本数，None表示全部测试（用于inference模式）")
    parser.add_argument("--inference_results_path", type=str, default=None, help="推理结果保存路径（用于inference模式）")
    
    args = parser.parse_args()
    
    # 构建分子配置字典
    mol_config = {
        'num_queries': args.num_queries,
        'input_dim': args.mol_input_dim,
        'num_heads': args.mol_num_heads
    }
    
    if args.mode == "train":
        # 统一训练入口
        logger.info(f"Starting training mode...")
        trained_model = train_sft_lora(
            data_path=args.data_path,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            max_seq_length=args.max_seq_length,
            epochs=args.epochs,
            lora_weights_path=args.lora_path,
            projector_path=args.projector_path,
            resume_from_checkpoint=args.resume_checkpoint,
            wandb_project=args.wandb_project,
            wandb_run_name=args.wandb_run_name,
            wandb_entity=args.wandb_entity,
            mol_config=mol_config,
            include_cot=args.include_cot,
        )
        logger.info("Training completed!")
    
    elif args.mode == "inference":
        # 推理模式
        logger.info(f"Starting inference mode...")
        # 如果没有指定 lora_path，默认从 output_dir 寻找
        lora_path = args.lora_path or os.path.join(args.output_dir, "lora_weights")
        projector_path = args.projector_path or os.path.join(args.output_dir, "projector.pt")
        
        # 确定测试数据路径
        test_data_path = args.test_data_path or args.data_path
        if not test_data_path:
            raise ValueError("Please specify test data path using --test_data_path or --data_path")
        
        # 确定结果保存路径
        if args.inference_results_path:
            results_path = args.inference_results_path
        else:
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
            results_path = os.path.join(args.output_dir, f"inference_results_{timestamp}.json")
        
        model, tokenizer = load_lora_model_for_inference(
            base_model_path=None, 
            lora_weights_path=lora_path,
            projector_path=projector_path,
            mol_config=mol_config
        )
        
        # 在测试数据上运行推理
        run_inference_on_test_data(
            model=model,
            tokenizer=tokenizer,
            test_data_path=test_data_path,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            save_results_path=results_path,
            max_samples=args.max_test_samples
        )
        
        logger.info("Inference completed!")
