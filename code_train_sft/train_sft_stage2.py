import torch
import torch.nn as nn
from trl import SFTTrainer, SFTConfig
from transformers import (
    AutoTokenizer, 
    TrainerCallback, 
    DataCollatorForSeq2Seq
)
from dataloader import load_data, extract_fields, llm_tokenize, coconut_tokenize
from model_new import Qwen3MoleculeLLM
import os
import time
import json
import logging
import wandb
import plotext as plt
from tqdm import tqdm
from config import ModelConfig
from typing import Dict, List, Any, Optional
from peft import LoraConfig, get_peft_model, TaskType, PeftModel, PeftConfig
from accelerate import Accelerator
import traceback
import tempfile


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

# 终端绘图回调：用于在控制台实时显示 Loss 曲线
class TerminalPlotCallback(TrainerCallback):
    def __init__(self):
        self.steps = []
        self.losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            # 只有主进程负责绘图
            if state.is_world_process_zero:
                self.steps.append(state.global_step)
                self.losses.append(logs["loss"])
                
                # 保持最近的 100 个点以保证终端显示效果
                display_steps = self.steps[-100:]
                display_losses = self.losses[-100:]

                # 终端绘图逻辑
                plt.clf()
                plt.plot(display_steps, display_losses, marker="dot", color="red", label="SFT Loss")
                plt.title("Real-time Training Loss (Terminal)")
                plt.xlabel("Step")
                plt.ylabel("Loss")
                
                # 设置合适终端大小的画布
                plt.plotsize(100, 25)
                plt.grid(True)
                plt.show()

# 工业级多模态数据整理器
class MultiModalDataCollator(DataCollatorForSeq2Seq):
    # ... (保持不变) ...
    def __call__(self, features):
        smiles = [f.pop("smiles") for f in features]
        batch = super().__call__(features)
        batch["smiles"] = smiles
        return batch

# 自定义 SFTTrainer 以支持多模态序列长度变化
class MultiModalSFTTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        重写 compute_loss。
        SFTTrainer 默认会在 compute_loss 中计算准确率指标，
        这要求 logits 和 labels 形状必须完全一致。
        在多模态场景下，序列会被拉长，因此我们回归标准 Trainer 的简单逻辑。
        
        注意：模型内部已经计算了正确的平均 Loss，这里直接返回即可。
        """
        if return_outputs:
            loss, outputs = super().compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
            return loss, outputs
        
        # 直接调用模型前向传播获取内部计算好的 Loss
        outputs = model(**inputs)
        
        # 模型已经返回正确的平均 Loss，直接使用
        return outputs.loss

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
    grad_accum=1,  # 新增：梯度累积步数
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
    
    # 初始化wandb (使用 mode="offline" 替代环境变量设置)
    wandb.init(
        project=wandb_project,
        name=wandb_run_name,
        entity=wandb_entity,
        mode="offline",
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
    # 无论是否加载了权重，我们都先确保基础模型是冻结的
    # 这是 PEFT 的标准实践：先全部冻结，再由 PEFT 开启特定层
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
        # 如果是加载的 PeftModel，PEFT 已经在 from_pretrained(..., is_trainable=True) 时处理好了梯度
        # 我们只需要确保它处于训练模式即可
        model.model.train()

    # 🚨 关键：只有 Projector 是我们需要手动处理的，因为它不在 LoRA 的管辖范围内
    logger.info("Ensuring projector is trainable...")
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
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        bf16=True,
        max_seq_length=max_seq_length,
        packing=False,
        dataset_text_field=None,
        remove_unused_columns=False,
        logging_steps=10,
        eval_strategy="no", # 不做划分，直接全部训练
        save_strategy="no", # 🚨 不保存中间检查点，节省空间
        save_total_limit=1,
        gradient_checkpointing=True, # 🚨 针对 8192 长度默认开启，防止 OOM
        max_grad_norm=0.3,
        warmup_ratio=0.1,
        report_to="wandb", # 🚨 开启官方 wandb 支持
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
    # 6. 初始化 MultiModalSFTTrainer
    # ============================
    logger.info("Initializing MultiModalSFTTrainer...")
    
    # 使用工业级整理器
    data_collator = MultiModalDataCollator(
        tokenizer=tokenizer,
        model=model.model, # 传入基础 LLM 模型以获取 Padding 配置
        padding=True,
    )
    
    trainer = MultiModalSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[LoraTrainingMonitorCallback(), TerminalPlotCallback()],
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

def load_lora_model_for_inference(
    base_model_path=None,
    lora_weights_path="./qwen3_mol_sft_lora_results/lora_weights",
    projector_path="./qwen3_mol_sft_lora_results/projector.pt",
    merge_lora=True,
    mol_config=None,
    accelerator: Accelerator = None,   # <- 新增：可选地传入 accelerate 实例
):
    """
    在 CPU 上安全地加载基础模型与 LoRA 权重（使用 main_process_first 避免重复 I/O），
    然后返回未移动到具体 GPU 的模型与 tokenizer。调用端应使用 accelerator.prepare(model)
    来把模型移动到每个进程对应的设备。

    如果传入了 accelerator，则使用 accelerator.main_process_first() 管理加载阶段。
    """
    if base_model_path is None:
        base_model_path = ModelConfig.DEFAULT_QWEN_PATH

    if mol_config is None:
        mol_config = {
            'num_queries': ModelConfig.NUM_QUERIES,
            'input_dim': ModelConfig.INPUT_DIM,
            'num_heads': ModelConfig.NUM_HEADS
        }

    logger.info(f"Preparing to load LoRA model from {base_model_path} with LoRA weights {lora_weights_path}")

    # 如果调用方没有传 accelerator，我们仍然会在本函数内创建一个（但推荐调用方传入）
    created_accelerator = False
    if accelerator is None:
        accelerator = Accelerator()
        created_accelerator = True

    # 在主进程先进行模型/权重文件的磁盘 I/O，防止 N 个进程同时读同一文件
    with accelerator.main_process_first():
        # 1) 在 CPU 上实例化基础模型（确保不会自动放到 GPU）
        # 如果 Qwen3MoleculeLLM 会将模型自动移动到 cuda，确保其构造/加载时使用 map_location='cpu' 或立刻 .to('cpu')
        model = Qwen3MoleculeLLM(qwen_model_name=base_model_path, mol_config=mol_config)
        tokenizer = model.tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # 强制移动到 CPU（以防某些构造会默认放到 GPU）
        model = model.to(torch.device("cpu"))

        # 2) 加载 LoRA 权重到 CPU（尽量避免直接落到单一 GPU）
        # 使用 device_map 或 map_location='cpu'，以确保权重先加载到 CPU，再交给 accelerate 分配到各自设备
        try:
            # 尝试使用 device_map 将权重加载到 CPU（PEFT / HF 兼容）
            model.model = PeftModel.from_pretrained(model.model, lora_weights_path, device_map={"": "cpu"})
        except Exception as e:
            logger.warning(f"PeftModel.from_pretrained with device_map failed: {e}; retrying with map to cpu via torch.load if available.")
            # 作为后备：尝试加载 state_dict 并手动加载（取决于你的 LoRA 权重存储形式）
            # 这里假定 lora_weights_path 是一个目录，PeftModel.from_pretrained 通常能处理；所以这只是兜底。
            model.model = PeftModel.from_pretrained(model.model, lora_weights_path, device_map={"": "cpu"})

        # 3) 可选：合并 LoRA（合并在 CPU 上进行，避免大量 GPU 内存占用）
        if merge_lora:
            logger.info("Merging LoRA weights on CPU for faster inference...")
            model.model = model.model.merge_and_unload()

        # 4) 加载 projector 权重到 CPU（如果存在）
        if os.path.exists(projector_path):
            projector_state_dict = torch.load(projector_path, map_location="cpu")
            model.projector.load_state_dict(projector_state_dict)
            model.projector = model.projector.to(torch.device("cpu"))
            logger.info(f"Loaded projector (CPU) from {projector_path}")

    # 等待所有进程完成主进程加载阶段
    accelerator.wait_for_everyone()

    # 返回模型（当前放在 CPU 上），调用者应使用 accelerator.prepare(model) 将它移动到各自设备
    # 如果我们在此函数内创建了 accelerator，需要提醒调用者他们没有传入 accelerator（但返回模型依然放 CPU）
    if created_accelerator:
        # 关闭本地 accelerator（可选）——通常更好的方式是由调用者统一管理 accelerator 生命周期
        # 这里不做额外处理，仅记录
        logger.debug("Note: load_lora_model_for_inference created an internal Accelerator; "
                     "it's recommended to create and pass Accelerator from caller for full control.")
    logger.info("Model loaded on CPU and ready for accelerator.prepare()")
    model.eval()
    return model, tokenizer

def load_test_data(test_data_path, max_len=None):
    """
    使用 dataloader.load_data 的 eval_mode 接口加载测试/验证数据（tokenized, labels=None）。

    如果 test_data_path 为目录：直接调用 load_data(path, eval_mode=True)
    如果 test_data_path 为单个文件：局部使用 load_dataset -> extract_fields(is_eval=True) -> llm_tokenize/coconut_tokenize 进行处理
    """

    if max_len is None:
        max_len = ModelConfig.MAX_TEXT_LEN

    logger.info(f"Loading test/eval data from: {test_data_path} (eval_mode=True)")

    # 情况1：目录（推荐）
    if os.path.isdir(test_data_path):
        # 直接使用 load_data 的 eval_mode
        dataset = load_data(test_data_path, include_cot=False, is_coconut=False, eval_mode=True, exclude_tasks=['rcr', 'mechsel'], max_len=max_len)
        logger.info(f"Loaded tokenized eval dataset from dir: {len(dataset)} examples")
        return dataset

# ------------------------------------------------------------
# 重写：运行推理，正确使用 accelerate 分配设备
# ------------------------------------------------------------
def run_inference_on_test_data(
    base_model_path=None,
    lora_weights_path="./qwen3_mol_sft_lora_results/lora_weights",
    projector_path="./qwen3_mol_sft_lora_results/projector.pt",
    merge_lora=True,
    test_data_path=None,
    max_new_tokens=2048,
    temperature=0.7,
    top_p=0.9,
    save_results_path=None,
    batch_size=1,
    max_samples=None,
    tokenization_max_len=None,
    mol_config=None,
    accelerator: Accelerator = None,   # <- 新增：可选地传入 accelerate 实例
):
    """
    使用 accelerate 在多进程（多 GPU）上做 sample-level 推理的主流程。
    关键点：
      - 在开始时创建 Accelerator；
      - 在主进程先加载模型与权重（CPU 上），然后使用 accelerator.prepare(model) 将模型移动到每个进程自己的设备。
      - 使用 accelerator.main_process_first() 与 accelerator.wait_for_everyone() 做 I/O 同步。
    """
    if test_data_path is None:
        raise ValueError("test_data_path must be provided")

    logger.info(f"Starting accelerated inference on {test_data_path}")
    if not accelerator:
        accelerator = Accelerator()
    proc_index = accelerator.process_index
    num_procs = accelerator.num_processes

    def aprint(*args, **kwargs):
        if accelerator.is_main_process:
            logger.info(" ".join(map(str, args)))
        else:
            logger.debug(f"proc{proc_index} - " + " ".join(map(str, args)))

    aprint(f"Accelerate process {proc_index}/{num_procs} started. Device: {accelerator.device}")

    # ---- 由 accelerator 管理的加载阶段：主进程先加载模型到 CPU，其他进程等待 ----
    model, tokenizer = load_lora_model_for_inference(
        base_model_path=base_model_path,
        lora_weights_path=lora_weights_path,
        projector_path=projector_path,
        merge_lora=merge_lora,
        mol_config=mol_config,
        accelerator=accelerator,   # 关键：把 accelerator 传入 load 函数以使用 main_process_first
    )

    # 把模型交给 accelerator 以将其移动到各自进程对应的设备（每个进程的 accelerator.device）
    # 注意：accelerator.prepare 也可用于 dataloader 等，但此处只对模型使用
    model = accelerator.prepare(model)

    # 确保 eval 模式
    model.eval()

    # 加载数据（使用你原先的 load_test_data）
    dataset = load_test_data(test_data_path, max_len=tokenization_max_len)
    if max_samples is not None and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))
    total_samples = len(dataset)
    aprint(f"Total eval samples: {total_samples}")

    # 分配索引
    assigned_indices = list(range(proc_index, total_samples, num_procs))
    aprint(f"Process {proc_index}: assigned {len(assigned_indices)} samples")

    local_results = []
    for local_i, idx in enumerate(tqdm(assigned_indices, desc=f"proc{proc_index}", disable=False)):
        item = dataset[idx]
        try:
            smiles_list = item.get("smiles", []) or []
            cleaned_smiles = [s.replace(".", "").strip() for s in smiles_list]

            input_ids = torch.tensor(item["input_ids"], dtype=torch.long).unsqueeze(0).to(accelerator.device)
            attention_mask = torch.tensor(item["attention_mask"], dtype=torch.long).unsqueeze(0).to(accelerator.device)

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

            gen0 = generated_ids[0]
            if isinstance(gen0, torch.Tensor):
                # 如果多进程需要跨进程收集，可以使用 accelerator.gather，但这里只需要把 tensor 转 CPU
                gen0_cpu = gen0.cpu()
                generated_text = tokenizer.decode(gen0_cpu, skip_special_tokens=True)
            else:
                try:
                    generated_text = tokenizer.decode(gen0, skip_special_tokens=True)
                except Exception:
                    generated_text = str(gen0)

            local_results.append({
                "sample_id": idx,
                "smiles": smiles_list,
                "result": generated_text.strip(),
                "task": item.get("task", None)
            })

        except Exception as e:
            logger.error(f"Error processing sample {idx} on proc {proc_index}: {e}")
            logger.error(traceback.format_exc())
            local_results.append({
                "sample_id": idx,
                "smiles": item.get("smiles", []),
                "result": None,
                "error": str(e),
                "task": item.get("task", None)
            })

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 每进程写部分文件
    if save_results_path:
        base_dir = os.path.dirname(save_results_path) if os.path.dirname(save_results_path) else "."
        os.makedirs(base_dir, exist_ok=True)
        part_prefix = os.path.join(base_dir, os.path.basename(save_results_path))
    else:
        tmpdir = tempfile.gettempdir()
        ts = int(time.time() * 1000)
        part_prefix = os.path.join(tmpdir, f"accelerate_inference_{ts}")

    part_path = f"{part_prefix}.part{proc_index}.json"
    with open(part_path, "w", encoding="utf-8") as pf:
        json.dump({
            "process_index": proc_index,
            "num_processes": num_procs,
            "local_results": local_results
        }, pf, indent=2, ensure_ascii=False)

    aprint(f"Process {proc_index} wrote partial results to {part_path}")

    # 等待所有进程写完
    accelerator.wait_for_everyone()

    # 主进程合并
    if accelerator.is_main_process:
        merged = []
        part_files = []
        for p in range(num_procs):
            candidate = f"{part_prefix}.part{p}.json"
            if os.path.exists(candidate):
                part_files.append(candidate)
            else:
                logger.warning(f"Expected part file missing: {candidate}")

        for pf in part_files:
            try:
                with open(pf, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    merged.extend(data.get("local_results", []))
            except Exception as e:
                logger.error(f"Failed to read part file {pf}: {e}")
                logger.error(traceback.format_exc())

        merged_sorted = sorted(merged, key=lambda x: x.get("sample_id", -1))

        save_file = save_results_path if save_results_path else f"{part_prefix}.merged.json"

        try:
            try:
                real_model = accelerator.unwrap_model(model)
            except Exception:
                real_model = model

            save_data = {
                "timestamp": datetime.now().isoformat(),
                "test_data_path": test_data_path,
                "model_info": {
                    "device": str(accelerator.device),
                    "total_parameters": sum(p.numel() for p in real_model.parameters()),
                    "trainable_parameters": sum(p.numel() for p in real_model.parameters() if p.requires_grad),
                },
                "generation_config": {
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                },
                "num_samples": len(merged_sorted),
                "test_results": merged_sorted
            }

            os.makedirs(os.path.dirname(save_file) if os.path.dirname(save_file) else ".", exist_ok=True)
            with open(save_file, "w", encoding="utf-8") as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Master process saved merged results to: {save_file}")
        except Exception as e:
            logger.error(f"Failed to save merged results: {e}")
            logger.error(traceback.format_exc())

        # 清理部分文件
        for pf in part_files:
            try:
                os.remove(pf)
            except Exception:
                pass

        logger.info(f"Inference completed on {len(merged_sorted)} samples (merged).")
        return merged_sorted
    else:
        return local_results

# 主函数 - 修改以支持第二轮训练
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LoRA微调多模态分子-语言模型")
    parser.add_argument("--mode", type=str, choices=["train", "inference"], default="train", help="运行模式")
    parser.add_argument("--output_dir", type=str, default=ModelConfig.DEFAULT_OUTPUT_DIR, help="保存/加载模型的路径")
    parser.add_argument("--data_path", type=str, default=ModelConfig.DEFAULT_DATA_PATH, help="数据路径")
    parser.add_argument("--batch_size", type=int, default=2, help="批次大小")
    parser.add_argument("--grad_accum", type=int, default=1, help="梯度累积步数")
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
            grad_accum=args.grad_accum,
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

        accelerator = Accelerator()

        # 在测试数据上运行推理（使用基于 load_data(..., eval_mode=True) 的流程）
        run_inference_on_test_data(
            base_model_path=None,
            lora_weights_path=lora_path,
            projector_path=projector_path,
            test_data_path=test_data_path,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            save_results_path=results_path,
            max_samples=args.max_test_samples,
            tokenization_max_len=min(args.max_seq_length, ModelConfig.MAX_TEXT_LEN),
            accelerator=accelerator,
        )

        logger.info("Inference completed!")
