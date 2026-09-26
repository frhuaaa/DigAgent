#!/usr/bin/env bash
#SBATCH --job-name=diagagent-worker
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --output=/finance_ML/huafengrui/DigAgent/slurm_logs/%x_%j.out
#SBATCH --error=/finance_ML/huafengrui/DigAgent/slurm_logs/%x_%j.err

set -Eeuo pipefail

readonly PROJECT_ROOT="/finance_ML/huafengrui/DigAgent"
readonly PYTHON_BIN="/home/huafengrui/anaconda3/envs/qlib/bin/python"
readonly SUBMIT_CONFIG="${1:?usage: sbatch diagagent_worker.slurm SUBMIT_CONFIG EXPERIMENT_ID}"
readonly EXPERIMENT_ID="${2:?usage: sbatch diagagent_worker.slurm SUBMIT_CONFIG EXPERIMENT_ID}"
readonly STAGE="${3:-experiment}"
readonly WORKER_PROFILE="${4:-training}"

cd "${PROJECT_ROOT}"

required_paths=(
  "${PYTHON_BIN}"
  "data/cn_data"
  "data/portfolio/c_2_c_1D.csv"
  "data/portfolio/mask_limit_up_1D.csv"
  "data/portfolio/mask_limit_down_1D.csv"
  "${SUBMIT_CONFIG}"
  "runs"
)

for required_path in "${required_paths[@]}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "FATAL: required path is missing: ${required_path}" >&2
    exit 2
  fi
done

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export CUBLAS_WORKSPACE_CONFIG=:4096:8

echo "job_id=${SLURM_JOB_ID}"
echo "node=$(hostname)"
echo "start_time=$(date --iso-8601=seconds)"
echo "submit=${SUBMIT_CONFIG}"
echo "experiment_id=${EXPERIMENT_ID}"
echo "stage=${STAGE}"
echo "worker_profile=${WORKER_PROFILE}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-none}"

if [[ "${WORKER_PROFILE}" != "portfolio" ]]; then
  if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "FATAL: Slurm did not assign CUDA_VISIBLE_DEVICES" >&2
    exit 3
  fi
  nvidia-smi
  "${PYTHON_BIN}" -c 'import qlib, torch; print("torch={} cuda_build={} cuda_available={} device_count={} qlib={}".format(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count(), getattr(qlib, "__version__", "unknown")))'
  REQUIRED_GPU_COUNT=1
  if [[ "${WORKER_PROFILE}" == "training" ]]; then
    REQUIRED_GPU_COUNT=3
  fi
  if ! "${PYTHON_BIN}" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() and torch.cuda.device_count() >= ${REQUIRED_GPU_COUNT} else 1)"; then
    echo "FATAL: ${REQUIRED_GPU_COUNT} allocated GPU(s) are not visible to PyTorch" >&2
    exit 4
  fi
fi

# Deliberately no .env and no Agent provider call on a compute node.
srun --ntasks=1 "${PYTHON_BIN}" -u scripts/run_execution_stage.py \
  --submit "${SUBMIT_CONFIG}" \
  --experiment-id "${EXPERIMENT_ID}" \
  --stage "${STAGE}"

echo "end_time=$(date --iso-8601=seconds)"
