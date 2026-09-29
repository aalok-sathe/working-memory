#!/bin/bash
#SBATCH --mail-user=chmosky@duck.com
#SBATCH --mail-type=ALL

#SBATCH --mem-per-cpu 64G

# Request a GPU partition node and access to 1 GPU
#SBATCH -p {slurm_partition_argument} --gres=gpu:1

#SBATCH -a 1-{array_upper_bound}%{array_upper_bound}
#SBATCH -t 1-01:00:00

#SBATCH -o {batch_output_prefix}batch-output/training_run_%A_%a.out
#SBATCH --nodes=1

set -x

. .venv/bin/activate
echo "find sample run at batch-output/training_run_${{SLURM_ARRAY_JOB_ID}}_1.out"

sleep $((RANDOM % 60 + 1))

# Pack {n_concurrent} concurrent training process(es) onto this one --gres=gpu:1
# allocation. SLURM's gres/gpu plugin sets CUDA_VISIBLE_DEVICES once for this job
# step; backgrounded children inherit it and therefore share the same physical GPU
# automatically -- no device-pinning needed. Each process independently pulls one
# unclaimed run off the shared wandb sweep queue.
{training_commands}
wait
