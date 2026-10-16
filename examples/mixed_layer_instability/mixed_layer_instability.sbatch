#!/bin/bash
#SBATCH --job-name=mli-demo
#SBATCH --partition=serc
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --output=mli-%j.out


module load openmpi/5.0.5

export JULIA_CUDA_MEMORY_POOL=none
export UCX_ERROR_SIGNALS="SIGILL,SIGBUS,SIGFPE"

srun -n 2 julia --project -O3 -- examples/mixed_layer_instability/mixed_layer_instability.jl
