## LatentX: A General Framework for Multimodal Latent Reasoning

### Clean up old LatentChem code
- [ ] better include/exclude task logics  
    no use. load_test_data has been per-domain. Not today
    - [ ] extract all include/exclude logic into bash script or yaml config
    - [ ] code
- [ ] clear training configs in inference entry
- [ ] refactor all chem-specified code
    - [ ] medium: passed config (TODO:M)  
        would completely change module names, scripts and more, so not today
        - [x] identify
        - [ ] code
    - [ ] weak: internal fields or variables (TODO:W)
        meaningless. Not today
        - [x] identify
        - [ ] code
