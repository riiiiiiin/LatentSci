#!/bin/bash

# Download and extract training dataset
echo "Downloading training dataset..."
huggingface-cli download --repo-type dataset anonymousssss22321/latentchem ChemCotDataset.tar.gz --local-dir .

echo "Extracting dataset..."
mkdir -p ChemCotDataset
tar -xzvf ChemCotDataset.tar.gz -C ChemCotDataset

echo "Running data cleaning..."
cd code_train_sft 
python xiufu.py
cd ..

