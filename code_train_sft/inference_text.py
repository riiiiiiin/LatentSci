import argparse
import logging
import os
import torch
from tqdm import tqdm
from peft import PeftModel
import json

# 假设这些是你项目中原本存在的模块
from config import ModelConfig
from model_text import load_qwen3_text_model

from inference import prepare_evaluation_dataset, save_inference_results

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================
# 2. 批量生成函数
# ==========================================
def batch_generate(model, tokenizer, batch_prompts, args, device, num_return_sequences=1):
    batch_input_texts = []
    for prompt in batch_prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        batch_input_texts.append(text)

    # 设为左填充以便生成
    tokenizer.padding_side = "left" 
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    inputs = tokenizer(
        batch_input_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_seq_length
    ).to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            num_return_sequences=num_return_sequences,
            do_sample=(args.temperature > 0 or num_return_sequences > 1),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    input_len = inputs.input_ids.shape[1]
    new_tokens = generated_ids[:, input_len:]
    decoded_responses = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
    
    return decoded_responses

# ==========================================
# 3. 主函数
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Batched Inference (Base Model or LoRA).")
    
    parser.add_argument("--base_model_path", type=str, default=ModelConfig.DEFAULT_QWEN_PATH)
    
    # 核心修改：去掉 required=True，默认 None
    parser.add_argument("--lora_path", type=str, default=None, help="Path to LoRA weights. If not provided, runs base model.")
    
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--inference_results_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--bf16", type=lambda x: (str(x).lower() == "true"), default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_test_samples", type=int, default=None, help="最大测试样本数，None表示全部测试（用于inference模式）")
    parser.add_argument("--include_tasks", type=str, nargs="*", default=None, help="需要包含的任务类型，None表示全部任务（用于inference模式）")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="每个prompt返回的生成序列数量")
    parser.add_argument("--proc_index", type=int, default=0,
                    help="当前进程索引 (0-based)，用于样本分片")
    parser.add_argument("--num_procs", type=int, default=1,
                        help="并行进程总数（样本分片数）")
    parser.add_argument("--gpu", type=int, default=None,
                        help="显卡 id，优先于 proc_index (如果提供则使用此 GPU)")

    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    num_return_sequences = args.num_return_sequences
    batch_size = args.batch_size
    # 1. 加载数据
    dataset, metadata, per_sample_metadata = prepare_evaluation_dataset(
        test_data_path=args.data_path,
        tokenization_max_len=min(args.max_seq_length, ModelConfig.MAX_TEXT_LEN),
        max_samples=args.max_test_samples,
        include_tasks=args.include_tasks,
        proc_index=args.proc_index,
        num_procs=args.num_procs,
        pure_text=True
    )
    # dataset.to_json("dataset.json")
    
    # 2. 加载 Base Model (这一步始终执行)
    logger.info(f"Loading Base Model from {args.base_model_path}...")
    model, tokenizer = load_qwen3_text_model(
        args.base_model_path,
        torch_dtype=(torch.bfloat16 if args.bf16 else torch.float32),
        device_map="auto",
    )

    # 3. 有条件地加载 LoRA
    if args.lora_path:
        if os.path.exists(args.lora_path):
            logger.info(f"Loading LoRA adapter from {args.lora_path}...")
            model = PeftModel.from_pretrained(model, args.lora_path)
        else:
            # 如果指定了路径但不存在，打印警告，并继续使用 Base Model
            logger.warning(f"Provided LoRA path '{args.lora_path}' does not exist! Proceeding with Base Model only.")
    else:
        logger.info("No LoRA path provided. Using Base Model only.")

    # 切换模式
    model.eval()
    model.config.use_cache = True
    
    prompts = dataset["query"]
    # 4. 批量推理循环
    generation_outputs = []
    
    print(device)
    logger.info(f"Starting inference (Batch Size: {batch_size})...")
    
    for i in tqdm(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[i : i + batch_size]
        decoded_list = batch_generate(model, tokenizer, batch_prompts, args, device, num_return_sequences)
        
        if num_return_sequences > 1:
            for i in range(batch_size):
                results = decoded_list[i * num_return_sequences : (i + 1) * num_return_sequences]
                generation_outputs.append({"results": [result.strip() for result in results]})
        else:     
            for decoded in decoded_list:
                generation_outputs.append({"result": decoded.strip()})
       
    print(len(generation_outputs), len(per_sample_metadata), len(prompts))         
    save_inference_results(
        save_results_path=args.inference_results_path,
        per_sample_metadata=per_sample_metadata,
        generation_outputs=generation_outputs,
        model=model,
        test_data_path=args.data_path,
        generation_config={
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p
        },
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    logger.info("Inference completed.")

if __name__ == "__main__":
    main()