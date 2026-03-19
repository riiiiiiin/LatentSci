from Chemistry.smi_ted_light.loadnew import load_smi_ted
from Chemistry.dataloader import load_data, load_grpo_data, load_test_data
from Chemistry.reward_func import (
    format_reward_answer_tag,
    reward_answer_correctness,
    reward_answer_correctness_bench,
    reward_answer_type_validity,
    reward_stage4_corrupt_or_correct,
    reward_stage4_double_scaled_correctness,
    reward_stage4_scaled_correctness,
)

registry = {
    'load_sci_embedder': load_smi_ted,
    'load_data': load_data,
    'load_grpo_data': load_grpo_data,
    'load_test_data': load_test_data,
    'reward_answer_correctness': reward_answer_correctness,
    'reward_answer_correctness_bench': reward_answer_correctness_bench,
    'reward_answer_type_validity': reward_answer_type_validity,
    'reward_stage4_corrupt_or_correct': reward_stage4_corrupt_or_correct,
    'reward_stage4_scaled_correctness': reward_stage4_scaled_correctness,
    'reward_stage4_double_scaled_correctness': reward_stage4_double_scaled_correctness,
    'format_reward_answer_tag': format_reward_answer_tag,
}