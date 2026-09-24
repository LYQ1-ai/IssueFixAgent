#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause
#
# M4 PRM 实机 GPU 执行脚本（在**宿主机 shell** 运行）。
#
# 为什么必须在宿主 shell 跑：DSH 的 bash 工具把每条命令都包在 bubblewrap 里，
# profile 硬编码 `--dev /dev`（新建只有 14 个节点的迷你 /dev），宿主 /dev/nvidia*
# 被结构性屏蔽，沙箱内 nvidia-smi 必然失败；详见 docs/prm_m4_runbook.md
# 「GPU 可用性现状」。本脚本不做任何环境假设，只按顺序调 PRM 的三个 CLI。
#
# 用法：
#   scripts/run_prm_m4_gpu.sh check     # GPU 自检 + 2 个 CUDA 门控单测（最先跑这个）
#   scripts/run_prm_m4_gpu.sh probe     # 步骤 2：M4.0 零训练侦察（§9.1，判据 AUC>0.55）
#   scripts/run_prm_m4_gpu.sh smoke     # 步骤 3：冒烟训练 20 steps @ 8K（§8）
#   scripts/run_prm_m4_gpu.sh train     # 步骤 4：正式训练（§8）
#   scripts/run_prm_m4_gpu.sh eval      # 步骤 5：评估（§9.2）
#   scripts/run_prm_m4_gpu.sh all       # check → probe → smoke → train → eval（中途失败即停）
#
# 环境变量（全部可覆盖）：
#   PRM_GPU=<物理卡号>       临时指定卡（等价于改 .env 的 CUDA_VISIBLE_DEVICES）；
#                            默认从仓库根 .env 读 CUDA_VISIBLE_DEVICES
#   PRM_RUN=m4-v1            训练 run 名（产物 outputs/prm/runs/<name>/）
#   PRM_PY=<python>          解释器：shell > .env 的 PRM_PY > 默认 <repo>/.venv-prm/bin/python
#   PRM_MIN_FREE_GB=20       训练启动前要求的最小空闲显存（不足则直接退出，不自动停服务）
#
# 日志：outputs/prm/logs/<step>-<时间戳>.log（同时打印到终端）
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# GPU 选择：优先 shell 已有的 CUDA_VISIBLE_DEVICES，其次仓库根 .env（见 prm/env.py）
env_get() {  # env_get KEY FILE → 值（去 export 前缀与引号）；找不到输出空
  [ -f "$2" ] || return 0
  sed -n "s/^[[:space:]]*\(export[[:space:]]\+\)\?$1=//p" "$2" | tail -1 \
    | sed 's/^["'"'"']//; s/["'"'"']$//'
}
if [ -n "${PRM_GPU:-}" ]; then
  export CUDA_VISIBLE_DEVICES="$PRM_GPU"
elif [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  _cvd="$(env_get CUDA_VISIBLE_DEVICES "$REPO/.env")"
  [ -n "$_cvd" ] && export CUDA_VISIBLE_DEVICES="$_cvd"
fi
PRM_GPU_IDX="${CUDA_VISIBLE_DEVICES%%,*}"; PRM_GPU_IDX="${PRM_GPU_IDX:-0}"
# 子进程 Python 输出不缓冲（否则 tqdm/print 会堵在 tee 管道里，看起来像卡死）
export PYTHONUNBUFFERED=1

PRM_DEVICE="cuda"   # 进程内可见卡 0 = .env 选中的那张（勿写物理卡号）

PRM_RUN="${PRM_RUN:-m4-v1}"
if [ -z "${PRM_PY:-}" ]; then
  PRM_PY="$(env_get PRM_PY "$REPO/.env")"          # 可用 .env 指定解释器（换环境训练）
fi
PRM_PY="${PRM_PY:-$REPO/.venv-prm/bin/python}"; export PRM_PY
PRM_MIN_FREE_GB="${PRM_MIN_FREE_GB:-20}"
CONFIG="${PRM_CONFIG:-config/prm.yaml}"
LOG_DIR="$REPO/outputs/prm/logs"
mkdir -p "$LOG_DIR"

log()  { printf '\n\033[1m=== %s ===\033[0m\n' "$*"; }
info() { printf '  %s\n' "$*"; }
die()  { printf '\n[FAIL] %s\n' "$*" >&2; exit 1; }

# 进程内可见卡号（cuda / cuda:0 → 0）；物理卡号见 PRM_GPU_IDX

# 单步执行并落日志：run_step <步骤名> <命令...>
run_step() {
  local name="$1"; shift
  local logf="$LOG_DIR/${name}-$(date +%Y%m%d-%H%M%S).log"
  log "步骤: $name  →  $*"
  info "日志: $logf"
  set +e
  "$@" 2>&1 | tee "$logf"
  local rc="${PIPESTATUS[0]}"
  set -e
  [ "$rc" -eq 0 ] || die "步骤 $name 失败（exit $rc），日志见 $logf"
  info "步骤 $name 完成 ✅"
}

require_gpu() {
  command -v nvidia-smi >/dev/null 2>&1 || die "找不到 nvidia-smi"
  log "GPU 自检"
  nvidia-smi --query-gpu=index,name,driver_version,memory.used,memory.total \
             --format=csv || die "nvidia-smi 失败（驱动或 /dev/nvidia* 不可用）"
  "$PRM_PY" - <<'PY' || die "torch 看不到 CUDA——先解决设备节点问题，再跑本脚本"
import sys, torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available(), "| devices:", torch.cuda.device_count())
sys.exit(0 if torch.cuda.is_available() else 1)
PY
}

# 训练前确认目标卡显存空闲（§1.2 GPU 纪律：不自动停任何服务）
check_free_mem() {
  local idx="$PRM_GPU_IDX"
  local free_gb
  free_gb="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$idx" \
             | awk '{printf "%.0f", $1/1024}')"
  info "物理卡 $idx 空闲显存 ${free_gb}GB（要求 >= ${PRM_MIN_FREE_GB}GB）"
  if [ "${free_gb:-0}" -lt "$PRM_MIN_FREE_GB" ]; then
    nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name --format=csv || true
    die "物理卡 $idx 空闲显存不足；改 .env 的 CUDA_VISIBLE_DEVICES（或 PRM_GPU=N）换卡，或与用户确认后再跑；脚本不会自动停服务"
  fi
}

require_data() {
  local missing=0
  for f in outputs/prm/train.parquet outputs/prm/dev.parquet; do
    [ -f "$f" ] || { printf '  缺少 %s\n' "$f" >&2; missing=1; }
  done
  [ "$missing" -eq 0 ] || die "数据产物不全——先跑 python -m prm.build_dataset --config $CONFIG"
}

step_check() {
  require_gpu
  # 训练侧不装 flash-attn 也能跑（train_prm 已按可用性自动降级 sdpa）
  run_step check-pytest "$PRM_PY" -m pytest test/test_prm_model.py -k TestRealModelGpu -v -rs
}

step_probe() {
  require_gpu
  require_data
  # 不传 --device：默认 cuda，即 .env 选中的那张卡
  run_step probe "$PRM_PY" -m prm.probe --config "$CONFIG"
  info "判据：报告里 AUC > 0.55 才继续；≈0.5 看 probe_report.md 里的备选 token 对"
}

_training() {  # _training <步骤名> <额外参数...>
  local name="$1"; shift
  require_gpu
  require_data
  check_free_mem
  # train_prm 没有 --device 参数：CUDA_VISIBLE_DEVICES 已在上面 export
  run_step "$name" "$PRM_PY" -m prm.train_prm --config "$CONFIG" --run "$PRM_RUN" "$@"
}

step_smoke() { PRM_RUN=smoke _training smoke --smoke; }

step_train() { _training "$PRM_RUN" ; }

step_eval() {
  require_gpu
  [ -d "outputs/prm/runs/$PRM_RUN" ] || die "找不到 outputs/prm/runs/$PRM_RUN —— 先跑 train"
  run_step eval "$PRM_PY" -m prm.eval_prm --config "$CONFIG" --run "$PRM_RUN"
}

usage() {
  sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'
}

case "${1:-}" in
  check) step_check ;;
  probe) step_probe ;;
  smoke) step_smoke ;;
  train) step_train ;;
  eval)  step_eval ;;
  all)   step_check; step_probe; step_smoke; step_train; step_eval ;;
  ""|-h|--help|help) usage ;;
  *) usage; die "未知步骤: $1" ;;
esac

log "全部完成 ✅ 产物：outputs/prm/{probe_report.*,runs/$PRM_RUN/,eval_report.*}；日志: $LOG_DIR"
