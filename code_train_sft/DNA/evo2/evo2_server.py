# evo2 requires torch==2.6.0 while vllm requires >= 2.9.0
# so we use conda for environment isolation and shared_tensor for low-cost ipc
# conda activate evo2_env
# export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
# TODO: the script should start this server seperately

import torch
from evo2 import Evo2

from shared_tensor import SharedTensorProvider, SharedTensorServer
provider = SharedTensorProvider(execution_mode="server")

layer_name = 'blocks.20.mlp.l3'


@provider.share(execution="direct", cache=True)
def load_evo2_model(model_name, model_path):
    global evo2_model
    if model_name != 'evo2_1b_base':
        raise NotImplementedError(
            f"Only evo2_1b_base is supported, found {model_name}"
        )
    evo2_model = Evo2(model_name, model_path)
    return

@provider.share(execution="direct", cache=False)
def get_dna_embedding(dna_sequence: str):
    global evo2_model

    input_ids = torch.tensor(
        evo2_model.tokenizer.tokenize(dna_sequence),
        dtype=torch.int,
    ).unsqueeze(0).to('cuda:0')
    outputs, embeddings = evo2_model(input_ids, return_embeddings=True, layer_names=[layer_name])
    return embeddings[layer_name][0]

@provider.share(execution="direct", cache=True)
def get_dna_embedding_batch(dna_sequences: list):
    global evo2_model
    pad_id = evo2_model.tokenizer.pad_id
    tokens_list = evo2_model.tokenizer.tokenize_batch(dna_sequences)
    
    lengths = [len(tokens) for tokens in tokens_list]
    max_len = max(lengths)
    batch_size = len(tokens_list)
    
    input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.int, device='cuda:0')
    
    for i, tokens in enumerate(tokens_list):
        input_ids[i, :lengths[i]] = torch.tensor(tokens, dtype=torch.int, device='cuda:0')
    
    outputs, embeddings = evo2_model(input_ids, return_embeddings=True, layer_names=[layer_name])
    full_embeddings = embeddings[layer_name]  # shape: (batch_size, max_len, hidden_dim)
    
    result = [full_embeddings[i, :lengths[i], :] for i in range(batch_size)]
    
    return result

server = SharedTensorServer(provider)
server.start(blocking=True)
