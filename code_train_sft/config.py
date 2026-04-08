import os

class ModelConfig:
    # --- 模型架构参数 (Architecture) ---
    INPUT_DIM = 768      # 分子编码器输出维度 (SMI-TED Light)
    NUM_QUERIES = 128    # 投影器的查询数量
    # OUTPUT_DIM 会在 model_new.py 中自动根据 LLM 维度适配
    NUM_HEADS = 8
    
    # --- 数据对齐参数 (Alignment) ---
    # 分子前缀总长度 = num_queries + <mol_start> + <mol_end>
    # 默认 128 + 2 = 130
    SMILES_LEN = NUM_QUERIES + 2 
    
    # --- 数据加载参数 (Data Loading) ---
    MAX_TEXT_LEN = 8192   # 文本部分的最大长度 (dataloader.py 中使用)
    
    # --- 默认路径 (Default Paths) ---
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Qwen 模型路径
    DEFAULT_QWEN_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "../models/Qwen3-8B-Base"))
    
    # Domain Identifier
    # This will be used in reflection_factory to get the correct sci model, dataloader and reward funcs
    DOMAIN = "Chemistry"
    VAL_SPLIT = False

domain_configs = {
    "Chemistry": {
        "sci_embedder_folder" : os.path.abspath(os.path.join(ModelConfig.CURRENT_DIR, "../models/smi-ted")),
        "sci_embedder_ckpt" : "smi-ted-Light_40.pt",
        "data_path" : os.path.abspath(os.path.join(ModelConfig.CURRENT_DIR, "../ChemCotDataset/chemcotbench-cot")),
        "output_dir" : os.path.join(ModelConfig.CURRENT_DIR, "qwen3_mol_sft_lora_results"),
        "val_split" : False
    },
    "DNA": {
        "sci_embedder_folder" : os.path.abspath(os.path.join(ModelConfig.CURRENT_DIR, "../models/evo2_1b_base")),
        "sci_embedder_ckpt" : "evo2_1b_base.pt",
        "data_path" : os.path.abspath(os.path.join(ModelConfig.CURRENT_DIR, "../data/kegg/data")),
        "output_dir" : os.path.join(ModelConfig.CURRENT_DIR, "qwen3_dna_sft_lora_results"),
        "val_split" : False
    },
}

ModelConfig.DEFAULT_SCI_EMBEDDER_FOLDER = domain_configs[ModelConfig.DOMAIN]["sci_embedder_folder"]
ModelConfig.DEFAULT_SCI_EMBEDDER_CKPT = domain_configs[ModelConfig.DOMAIN]["sci_embedder_ckpt"]
ModelConfig.DEFAULT_DATA_PATH = domain_configs[ModelConfig.DOMAIN]["data_path"]
ModelConfig.DEFAULT_OUTPUT_DIR = domain_configs[ModelConfig.DOMAIN]["output_dir"]
ModelConfig.VAL_SPLIT = domain_configs[ModelConfig.DOMAIN]["val_split"]