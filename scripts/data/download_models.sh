#!/bin/bash

# Download base models
echo "Downloading Qwen3-8B-Base..."
huggingface-cli download --resume-download Qwen/Qwen3-8B-Base --local-dir ./models/Qwen3-8B-Base

echo "Downloading smi-ted..."
huggingface-cli download --resume-download ibm-research/materials.smi-ted --local-dir ./models/smi-ted

