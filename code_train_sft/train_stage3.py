import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, TrainingArguments, TrainerCallback
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import logging
from datetime import datetime
import argparse
import wandb
import plotext as plt

# 导入我们的自定义组件
from model_stage3 import Qwen3MoleculeLLM
from dataloader import load_data, COCONUT_TOKENS
from config import ModelConfig
from train_sft_stage2 import MultiModalDataCollator, MultiModalSFTTrainer, LoraTrainingMonitorCallback, TerminalPlotCallback

def load_trained_components_stage3(model, lora_weights_path=None, mm_projector_path=None):
    """
    Unified loader for Stage 3: loads LoRA weights and combined Projector + Bio Updater weights.
    Assumes unified multi-modal checkpoint format.
    """
    # 1. 加载 LoRA 权重
    if lora_weights_path and os.path.exists(lora_weights_path):
        logger.info(f"Loading LoRA weights from: {lora_weights_path}")
        model.model = PeftModel.from_pretrained(
            model.model, 
            lora_weights_path,
            is_trainable=True 
        )
    
    # 2. 加载多模态组合权重 (Projector + Bio Updater)
    if mm_projector_path and os.path.exists(mm_projector_path):
        logger.info(f"Loading unified multi-modal weights from: {mm_projector_path}")
        device = next(model.parameters()).device
        checkpoint = torch.load(mm_projector_path, map_location=device)
        
        # 直接按组合格式加载
        model.projector.load_state_dict(checkpoint['projector'])
        logger.info("Successfully loaded both projector and bio_updater.")
            
    return model

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_stage3():
    parser = argparse.ArgumentParser(description="Stage 3 Training for Bio-LatentCOT")
    parser.add_argument("--data_path", type=str, default="/mnt/afs/L202500070/Bio-LatentCOT/ChemCotDataset/chemcotbench-cot")
    parser.add_argument("--lora_path", type=str, default=None, help="Stage 2 LoRA weights (optional)")
    parser.add_argument("--projector_path", type=str, default=None, help="Unified projector + bio_updater weights (optional)")
    parser.add_argument("--output_dir", type=str, default="./outputs/stage3_coconut")
    parser.add_argument("--epochs_per_stage", type=int, default=3, help="Number of epochs to train (per latent stage or for SFT)")
    parser.add_argument("--max_latent_stage", type=int, default=3, help="Max number of CoT steps to latent-ize")
    parser.add_argument("--c_thought", type=int, default=2, help="Number of latent tokens per CoT step")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_seq_length", type=int, default=8192)
    parser.add_argument("--save_full_model", type=lambda x: (str(x).lower() == 'true'), default=False, help="Whether to save full model weights (default False to save space)")
    parser.add_argument("--training_stage", type=int, default=3, choices=[1, 2, 3], help="Which stage to train: 1 (No COT), 2 (With COT), 3 (Latent/Coconut)")
    # Counterfactual bio-token embedding perturbation + loss
    parser.add_argument("--cf_lambda", type=float, default=0.0, help="Weight for counterfactual hinge loss (0 disables).")
    parser.add_argument("--cf_margin", type=float, default=0.5, help="Margin for hinge on (L_cf - L_pos).")
    parser.add_argument("--cf_prob", type=float, default=1.0, help="Probability of triggering a counterfactual paired-loss pass for a batch.")
    
    args = parser.parse_args()

    # 1. 基础配置
    mol_config = {
        'num_queries': ModelConfig.NUM_QUERIES,
        'input_dim': ModelConfig.INPUT_DIM,
        'num_heads': ModelConfig.NUM_HEADS
    }
    
    # 当前的权重路径，初始为参数传入的路径
    current_lora_path = args.lora_path
    current_projector_path = args.projector_path

    # 2. 开启训练循环
    # Stage 1 & 2 只训练一次，Stage 3 开启分阶段循环训练
    if args.training_stage == 1:
        stages = [0]
        is_coconut = False
        include_cot = False
        mode_name = "Stage1-NoCOT"
    elif args.training_stage == 2:
        stages = [0]
        is_coconut = False
        include_cot = True
        mode_name = "Stage2-WithCOT"
    else: # Stage 3
        stages = range(args.max_latent_stage + 1)
        is_coconut = True
        include_cot = True
        mode_name = "Stage3-Coconut"

    for stage in stages:
        logger.info(f"\n" + "🚀" * 30)
        logger.info(f"STARTING {mode_name} (STAGE {stage})")
        if is_coconut:
            logger.info(f"Replace first {stage} steps with {stage * args.c_thought} latents")
        logger.info("🚀" * 30 + "\n")

        # 2.1 每一个 Stage 彻底重新初始化模型
        model = Qwen3MoleculeLLM(qwen_model_name=ModelConfig.DEFAULT_QWEN_PATH, mol_config=mol_config)
        tokenizer = model.tokenizer
        
        # 加载上一个 Stage 的权重
        if current_lora_path or current_projector_path:
            logger.info(f"Loading weights for training...")
            model = load_trained_components_stage3(
                model, 
                lora_weights_path=current_lora_path, 
                mm_projector_path=current_projector_path
            )
        
        # 确保 LoRA 已配置
        if not hasattr(model.model, 'peft_config') or model.model.peft_config is None:
            logger.info("Configuring LoRA from scratch...")
            # 首先冻结所有参数
            for param in model.parameters():
                param.requires_grad = False
            
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=16,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                bias="none",
            )
            model.model = get_peft_model(model.model, lora_config)
        
        # 确保投影器和 Bio Updater 可训练
        for param in model.projector.parameters():
            param.requires_grad = True
        
        for param in model.bio_updater.parameters():
            param.requires_grad = True
        
        model.model.train()

        # 2.2 重新加载当前 Stage 的数据集
        train_dataset = load_data(
            args.data_path,
            include_cot=include_cot,
            is_coconut=is_coconut,
            scheduled_stage=stage,
            c_thought=args.c_thought,
            max_len=args.max_seq_length
        )

        # 2.3 配置当前 Stage 的输出目录
        stage_suffix = f"stage{args.training_stage}_sub{stage}" if is_coconut else f"stage{args.training_stage}"
        stage_output_dir = os.path.join(args.output_dir, stage_suffix)
        
        training_args = SFTConfig(
            output_dir=stage_output_dir,
            num_train_epochs=args.epochs_per_stage, # 每个阶段练固定 Epoch
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            bf16=True,
            max_seq_length=args.max_seq_length,
            remove_unused_columns=False,
            logging_steps=10,
            save_strategy="no", # 🚨 不保存中间检查点，节省空间
            save_total_limit=1,
            gradient_checkpointing=True,
            # 🚨 修复 DDP 错误：使用非重入式 checkpoint 并允许查找未使用参数
            gradient_checkpointing_kwargs={"use_reentrant": False},
            ddp_find_unused_parameters=True,
            report_to="wandb",
            optim="adamw_8bit",
            lr_scheduler_type="cosine",
            weight_decay=0.01,
        )

        # WandB 记录
        if wandb.run is not None:
            wandb.finish() # 结束上一个 stage 的 run
        
        wandb_run_name = f"{mode_name}-sub{stage}" if is_coconut else mode_name
        wandb.init(
            project="qwen3-molecule-unified",
            name=f"{wandb_run_name}-{datetime.now().strftime('%m%d-%H%M')}",
            mode="offline",
            config={**vars(args), "current_stage": stage, "mode": mode_name}
        )

        data_collator = MultiModalDataCollator(tokenizer=tokenizer, model=model.model, padding=True)

        trainer = MultiModalSFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            cf_lambda=args.cf_lambda,
            cf_margin=args.cf_margin,
            cf_prob=args.cf_prob,
            callbacks=[LoraTrainingMonitorCallback(), TerminalPlotCallback()],
        )

        # 执行当前阶段训练
        trainer.train()
        
        # 2.4 保存当前阶段结果，并更新下个阶段的加载路径
        if args.save_full_model:
            logger.info("Saving full model weights for stage %s...", stage_suffix)
            trainer.save_model(stage_output_dir)
        else:
            logger.info("Skipping full model weights saving for stage %s (only saving LoRA and Projector).", stage_suffix)
            
        current_lora_path = os.path.join(stage_output_dir, "lora_weights")
        current_projector_path = os.path.join(stage_output_dir, "mm_projector.pt")
        
        # 手动保存 LoRA 和 组合后的多模态权重
        os.makedirs(current_lora_path, exist_ok=True)
        model.model.save_pretrained(current_lora_path)
        
        # 将 Projector 和 Bio Updater 存入同一个文件
        mm_weights = {
            'projector': model.projector.state_dict(),
            'bio_updater': model.bio_updater.state_dict()
        }
        torch.save(mm_weights, current_projector_path)
        
        # 如果不保存全模型，我们也至少存一下分词器，方便后续推理加载
        tokenizer.save_pretrained(stage_output_dir)
        
        logger.info(f"✅ {mode_name} Stage {stage} completed. Weights saved to {stage_output_dir}")
        
        # 显存清理
        del trainer, model, train_dataset
        torch.cuda.empty_cache()

    logger.info(f"🎉 All {mode_name} Stages completed!")

if __name__ == "__main__":
    train_stage3()

