#!/usr/bin/env bash
# Target-specific compatibility worker. Submit through scripts/run_controller.py.
# Usage: sbatch slurm/lstm_csi1000_deepseek.slurm EXP_001
#SBATCH --job-name=diagagent-csi1000
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=/finance_ML/huafengrui/DigAgent/slurm_logs/%x_%j.out
#SBATCH --error=/finance_ML/huafengrui/DigAgent/slurm_logs/%x_%j.err

set -Eeuo pipefail

readonly PROJECT_ROOT="/finance_ML/huafengrui/DigAgent"
readonly PYTHON_BIN="/home/huafengrui/anaconda3/envs/qlib/bin/python"
readonly SUBMIT_CONFIG="submit/lstm_csi1000_deepseek.json"
readonly EXPERIMENT_ID="${1:?usage: sbatch lstm_csi1000_deepseek.slurm EXPERIMENT_ID}"

cd "${PROJECT_ROOT}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "FATAL: Slurm did not assign CUDA_VISIBLE_DEVICES" >&2
  exit 3
fi

# Deliberately no .env and no Agent provider call on a compute node.
srun --ntasks=1 "${PYTHON_BIN}" -u scripts/run_execution_stage.py \
  --submit "${SUBMIT_CONFIG}" \
  --experiment-id "${EXPERIMENT_ID}"
