#!/usr/bin/env bash
#SBATCH --job-name=diagagent-researcher
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/finance_ML/huafengrui/DigAgent/slurm_logs/%x_%j.out
#SBATCH --error=/finance_ML/huafengrui/DigAgent/slurm_logs/%x_%j.err

set -Eeuo pipefail

readonly PROJECT_ROOT="/finance_ML/huafengrui/DigAgent"
readonly PYTHON_BIN="/home/huafengrui/anaconda3/envs/qlib/bin/python"
readonly RUN_ROOT="${1:-runs/lstm_csi1000_deepseek}"

cd "${PROJECT_ROOT}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "FATAL: Slurm did not assign CUDA_VISIBLE_DEVICES" >&2
  exit 3
fi

"${PYTHON_BIN}" -u scripts/rerun_researcher_outputs.py --run-root "${RUN_ROOT}"
