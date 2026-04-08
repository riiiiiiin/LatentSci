# dependencies
needs gpu, needs 22.04
```
conda create -n evo2_dev python=3.12
pip install --upgrade pip setuptools wheel
pip install torch==2.6.0
conda install -c nvidia cuda-nvcc cuda-cudart-dev
conda install -c conda-forge transformer-engine-torch=2.3.0
pip install psutil
pip install flash-attn==2.8.0.post2 --no-build-isolation
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
pip install evo2
```

# dev2: evo2 backup
```
conda activate latentsci-dna-dev2
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

# evo2_env
```
conda activate evo2_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```