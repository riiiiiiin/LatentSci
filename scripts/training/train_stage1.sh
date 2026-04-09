#!/bin/bash

source /mnt/afs/L202500070/miniconda3/etc/profile.d/conda.sh
LOG_FILE="evo2_server_output.log"

rm "$LOG_FILE"
conda activate evo2_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
python "code_train_sft/DNA/evo2/evo2_server.py" > $LOG_FILE 2>&1 &
SERVER_PID=$!


for i in {1..30}; do
    if grep -q "\[shared_tensor\] INFO shared_tensor.server: Server listening" $LOG_FILE; then
        echo "Found server listening message"
        break
    fi
    echo "Waiting for evo2 shared-tensor server to start... (attempt $i/30)"
    sleep 10
done

if ! grep -q "\[shared_tensor\] INFO shared_tensor.server: Server listening" $LOG_FILE; then
    echo "Server listening message not found after 30 attempts"
    kill $SERVER_PID
    exit 1
fi

conda deactivate
conda activate latentsci-dna
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Stage 1: Initial Training
echo "Starting Training Stage 1..."
cd code_train_sft 
accelerate launch --multi_gpu --num_processes 8 train_stage3.py \
  --training_stage 1 \
  --epochs_per_stage 3 \
  --output_dir ./outputs/stage1__lr2e-4__cf_lambda_0.2__cf_margin_0.1__cf_prob_1.0 \
  --batch_size 4 \
  --grad_accum 1 \
  --lr 2e-4 \
  --cf_lambda 0.2 --cf_margin 0.1 \
  --cf_prob 1.0

cd ..

echo "Killing evo2 server"
kill $SERVER_PID