#!/bin/bash
#SBATCH --job-name=jit-soflow
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --gpus-per-node=4
#SBATCH --output=logs/jit-soflow_%j.out
#SBATCH --error=logs/jit-soflow_%j.out

echo "=========================================="
echo "Job ID:      $SLURM_JOB_ID"
echo "Node:        $(hostname)"
echo "Partition:   $SLURM_JOB_PARTITION"
echo "GPUs:        ${SLURM_GPUS_ON_NODE:-N/A}"
echo "Start:       $(date)"
echo "=========================================="

source /users/rradev/JiT/.venv/bin/activate

torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 \
  main_jit.py \
  --model JiT-B/4 \
  --proj_dropout 0.0 \
  --P_mean -0.8 --P_std 0.8 --class_num 102 \
  --label_drop_prob 1.0 --pin_mem \
  --img_size 64 --noise_scale 1.0 \
  --batch_size 4 --blr 5e-3 \
  --epochs 600 --warmup_epochs 5 \
  --gen_bsz 32 --num_images 102 --cfg 1.0 --interval_min 0.1 --interval_max 1.0 \
  --output_dir /users/rradev/JiT/output_soflow \
  --wandb_run_name soflow \
  --soflow --lambda_fm 0.75 --r_init 0.1 --r_end 0.002 --r_total_steps 100000 \
  --online_eval

echo "=========================================="
echo "End:         $(date)"
echo "=========================================="
