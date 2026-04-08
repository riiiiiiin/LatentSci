from DNA.evo2.evo2_client import load_evo2
from DNA.dataloader import load_data, load_grpo_data, load_test_data

registry = {
    'load_sci_embedder': load_evo2,
    'load_data': load_data,
    'load_grpo_data': load_grpo_data,
    'load_test_data': load_test_data
}