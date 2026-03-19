## LatentX: A General Framework for Multimodal Latent Reasoning

### Clean up old LatentChem code
- [ ] better include/exclude task logics
    - [ ] extract all include/exclude logic into bash script or yaml config
    - [ ] code
- [ ] clear training configs in inference entry
- [ ] refactor all chem-specified code
    - [ ] strong: model-specific + passed prompt key (TODO:S)
        - [x] identify
        - [ ] code
            - [x] SciEmbedder has to implement parameters(), eval() and encode()
            - [x] DataLoader has to implement load_data(), load_test_data() and load_grpo_data()
            - [x] reward funcs has to implement format_reward_answer_tag(), reward_answer_correctness(), reward_answer_correctness_bench(), reward_answer_type_validity(), reward_stage4_corrupt_or_correct(), reward_stage4_double_scaled_correctness(), reward_stage4_scaled_correctness
        - [ ] test
    - [ ] medium: passed config (TODO:M)
        - [x] identify
        - [ ] code
    - [ ] weak: internal fields or variables (TODO:W)
        - [x] identify
        - [ ] code
