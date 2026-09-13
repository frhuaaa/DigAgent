#!/usr/bin/env bash
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
readonly SUBMIT_CONFIG="submit/lstm_csi1000_gpt.json"

cd "${PROJECT_ROOT}"
mkdir -p slurm_logs

required_paths=(
  "${PYTHON_BIN}"
  ".env"
  "data/cn_data"
  "data/portfolio/c_2_c_1D.csv"
  "data/portfolio/mask_limit_up_1D.csv"
  "data/portfolio/mask_limit_down_1D.csv"
  "${SUBMIT_CONFIG}"
)

for required_path in "${required_paths[@]}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "FATAL: required path is missing: ${required_path}" >&2
    exit 2
  fi
done

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "FATAL: Slurm did not assign CUDA_VISIBLE_DEVICES" >&2
  exit 3
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "job_id=${SLURM_JOB_ID}"
echo "job_name=${SLURM_JOB_NAME}"
echo "node=$(hostname)"
echo "start_time=$(date --iso-8601=seconds)"
echo "project_root=${PROJECT_ROOT}"
echo "python=${PYTHON_BIN}"
echo "submit=${SUBMIT_CONFIG}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"

nvidia-smi
"${PYTHON_BIN}" -c 'import qlib, torch; print("torch={} cuda_build={} cuda_available={} device_count={} qlib={}".format(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count(), getattr(qlib, "__version__", "unknown")))'

if ! "${PYTHON_BIN}" -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() and torch.cuda.device_count() >= 1 else 1)'; then
  echo "FATAL: allocated GPU is not visible to PyTorch" >&2
  exit 4
fi

srun --ntasks=1 "${PYTHON_BIN}" -u scripts/run_adaptation.py \
  --submit "${SUBMIT_CONFIG}"

echo "end_time=$(date --iso-8601=seconds)"
