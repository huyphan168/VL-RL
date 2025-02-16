#!/bin/bash

#SBATCH --job-name=TrainGP
#SBATCH --mail-user=user@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --nodes=1                                 # Total number of nodes requested
#SBATCH --ntasks-per-node=8                      # Total number of task requested
#SBATCH --cpus-per-task=8                        # Total number of cores requested
#SBATCH --mem=1000G
#SBATCH --time=96:00:00                     # Time limit (hh:mm:ss)
#SBATCH --gpus-per-node=8   
#SBATCH --output=./slurm_logs/train_%A_%a.out
#SBATCH --output=./slurm_logs/train_%A_%a.err
#SBATCH --partition=gpu

# LR=1e-6
save_every=1
save_model=False # disable running saving. one checkpoint ~30GB

# CKPT_NAME="tianzhechu/GP-VL-Init" # official init model: tianzhechu/GP-VL-Init
PORT=$((RANDOM % 10000 + 1000))

DS_SKIP_CUDA_CHECK=1 TOKENIZERS_PARALLELISM=false accelerate launch \
    --config_file scripts/config_zero2_8gpu.yaml \
    --main_process_port ${PORT} -m rl.launcher \
    -f rl/configs/llama_gp_vl_liger.yaml \
    --output_dir=train_ckpt/gp_vl/ \
    --optimizer_config.lr_max_steps=100 \
    --prompt_config.enable_verification=True \
    --num_updates=20 \
    --env_config.treat_face_cards_as_10=True \
    --env_config.face_cards_color=black \
    --run_name=gp_vl_training \
    --save_ckpt=${save_model} \
    --save_every=${save_every} \
    # --optimizer_config.init_lr=${LR} \
    # --model_path=${CKPT_NAME} \
