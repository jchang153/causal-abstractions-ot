#!/usr/bin/env bash
#SBATCH --job-name=binadd-10seed
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=4
#SBATCH --mem=128g
#SBATCH --time=48:00:00
#SBATCH --constraint="scratch"
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --requeue

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
export PYTHONUNBUFFERED=1
export WORKERS=4
export REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR}}"

bash experiments/binary_addition/slurm/run_delta_binary_addition_10seed_suite.sh
