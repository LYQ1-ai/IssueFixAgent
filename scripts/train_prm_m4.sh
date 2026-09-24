#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause
#
# M4 正式训练脚本（在**宿主机 shell** 运行；DSH 沙箱内无 /dev/nvidia*，必失败）。
#
# 流程：预检 → （可选）probe 门禁 → 冒烟 20 steps @ 8K → 正式训练 → 评估
#   · 预检：GPU/显存、数据产物、template_hash 一致性、磁盘空间、run 目录占用
#   · 断点续训：--resume 从 outputs/prm/runs/<run>/checkpoints/ 最新检查点继续
#   · 全程日志落 outputs/prm/logs/，可 --background 丢到 nohup 后台
#
# 用法：
#   scripts/train_prm_m4.sh                      # GPU0，跑 smoke → m4-v1 → eval
#   scripts/train_prm_m4.sh --run m4-v1          # 指定 run 名
#   scripts/train_prm_m4.sh --resume             # 训练中断后接着跑
#   scripts/train_prm_m4.sh --skip-smoke         # 跳过冒烟
#   scripts/train_prm_m4.sh --no-eval            # 只训练不评估
#   scripts/train_prm_m4.sh --require-probe      # probe_report.json 未过门禁则中止
#   scripts/train_prm_m4.sh --gpu 1              # 临时换卡（默认读 .env 的 CUDA_VISIBLE_DEVICES）
#   换解释器：PRM_PY=/path/to/python ./scripts/train_prm_m4.sh（或写进 .env 的 PRM_PY）
#   scripts/train_prm_m4.sh --background         # nohup 后台跑，打印日志路径
#   scripts/train_prm_m4.sh --dry-run            # 只打印将要执行的命令
#
# 兼容旧入口：scripts/run_prm_m4_gpu.sh（check/probe/smoke/train/eval/all 分步）
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

RUN="m4-v1"
DEVICE="cuda"        # 进程内可见卡（.env 选中的那张）；勿写物理卡号
# 子进程 Python 输出不缓冲（否则 tqdm/print 会堵在 tee 管道里，看起来像卡死）
export PYTHONUNBUFFERED=1

PRM_GPU=""          # 物理卡号覆盖（等价于改 .env 的 CUDA_VISIBLE_DEVICES）
RESUME=0
SKIP_SMOKE=0
NO_EVAL=0
REQUIRE_PROBE=0
DRY_RUN=0
BACKGROUND=0
MIN_FREE_GB=30
MAX_LENGTH=""
FORCE=0
CONFIG="${PRM_CONFIG:-config/prm.yaml}"
LOG_DIR="$REPO/outputs/prm/logs"

while [ $# -gt 0 ]; do
  case "$1" in
    --run)            RUN="$2"; shift 2 ;;
    --device)         DEVICE="$2"; shift 2 ;;
    --gpu)            PRM_GPU="$2"; shift 2 ;;
    --resume)         RESUME=1; shift ;;
    --skip-smoke)     SKIP_SMOKE=1; shift ;;
    --no-eval)        NO_EVAL=1; shift ;;
    --require-probe)  REQUIRE_PROBE=1; shift ;;
    --dry-run)        DRY_RUN=1; shift ;;
    --background)     BACKGROUND=1; shift ;;
    --force)          FORCE=1; shift ;;
    --min-free-gb)    MIN_FREE_GB="$2"; shift 2 ;;
    --max-length)     MAX_LENGTH="$2"; shift 2 ;;
    --config)         CONFIG="$2"; shift 2 ;;
    -h|--help)        awk 'NR>1 && /^#/ {sub(/^# ?/,""); print; next} NR>1 {exit}' "$0"; exit 0 ;;
    *) echo "未知参数: $1（用 --help 查看用法）" >&2; exit 2 ;;
  esac
done

# GPU 选择：--gpu 覆盖 > shell 环境 > 仓库根 .env（见 prm/env.py），不再写死卡号
env_get() {  # env_get KEY FILE → 值（去 export 前缀与引号）
  [ -f "$2" ] || return 0
  sed -n "s/^[[:space:]]*\(export[[:space:]]\+\)\?$1=//p" "$2" | tail -1 \
    | sed 's/^["'"'"']//; s/["'"'"']$//'
}
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -n "$PRM_GPU" ]; then
  export CUDA_VISIBLE_DEVICES="$PRM_GPU"
elif [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  _cvd="$(env_get CUDA_VISIBLE_DEVICES "$REPO_ROOT/.env")"
  [ -n "$_cvd" ] && export CUDA_VISIBLE_DEVICES="$_cvd"
fi
GPU_IDX="${CUDA_VISIBLE_DEVICES%%,*}"; GPU_IDX="${GPU_IDX:-0}"   # 物理卡号（nvidia-smi 用）

# 解释器：shell 的 PRM_PY > .env 的 PRM_PY > 默认 .venv-prm > PATH 里的 python
# （换 conda/新 venv 训练时，把 PRM_PY 写进 .env 即可，无需改脚本）
if [ -n "${PRM_PY:-}" ]; then
  PY="$PRM_PY"
else
  _py="$(env_get PRM_PY "$REPO_ROOT/.env")"; PY="${_py:-}"
fi
[ -n "$PY" ] || { [ -x "$REPO/.venv-prm/bin/python" ] && PY="$REPO/.venv-prm/bin/python" || PY="$(command -v python)"; }

mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d-%H%M%S)"
MAIN_LOG="$LOG_DIR/train-${RUN}-${TS}.log"

log()  { printf '\n\033[1m=== %s ===\033[0m\n' "$*"; }
info() { printf '  %s\n' "$*"; }
die()  { printf '\n[FAIL] %s\n' "$*" >&2; exit 1; }
run_or_echo() { if [ "$DRY_RUN" -eq 1 ]; then printf '  [dry-run] %s\n' "$*"; else "$@"; fi; }

# ---------------------------------------------------------------- 后台模式
if [ "$BACKGROUND" -eq 1 ] && [ "${PRM_TRAIN_BG:-0}" != "1" ]; then
  ARGS=()
  [ "$RESUME" -eq 1 ] && ARGS+=(--resume)
  [ "$SKIP_SMOKE" -eq 1 ] && ARGS+=(--skip-smoke)
  [ "$NO_EVAL" -eq 1 ] && ARGS+=(--no-eval)
  [ "$REQUIRE_PROBE" -eq 1 ] && ARGS+=(--require-probe)
  [ "$FORCE" -eq 1 ] && ARGS+=(--force)
  [ -n "$PRM_GPU" ] && ARGS+=(--gpu "$PRM_GPU")
  ARGS+=(--run "$RUN" --device "$DEVICE" --min-free-gb "$MIN_FREE_GB" --config "$CONFIG")
  [ -n "$MAX_LENGTH" ] && ARGS+=(--max-length "$MAX_LENGTH")
  PRM_TRAIN_BG=1 nohup bash "$0" "${ARGS[@]}" > "$MAIN_LOG" 2>&1 &
  PID=$!
  printf '\n已在后台启动（PID %s）\n  日志: %s\n  跟踪: tail -f %s\n  停止: kill %s\n' \
         "$PID" "$MAIN_LOG" "$MAIN_LOG" "$PID"
  exit 0
fi

# ---------------------------------------------------------------- 预检
log "预检 · 运行环境"
info "python : $PY"
info "config : $CONFIG"
info "run    : $RUN（输出 outputs/prm/runs/$RUN）"
info "GPU    : $DEVICE（CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-未设置} → 物理卡 $GPU_IDX；显存门槛 ${MIN_FREE_GB}GB）"
[ "$RESUME" -eq 1 ] && info "续训   : 开启" || true

command -v nvidia-smi >/dev/null 2>&1 || die "找不到 nvidia-smi"
nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv -i "$GPU_IDX" \
  || die "物理卡 $GPU_IDX 不可用（nvidia-smi 失败）"
FREE_GB="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU_IDX" \
           | awk '{printf "%.0f", $1/1024}')"
if [ "${FREE_GB:-0}" -lt "$MIN_FREE_GB" ]; then
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name --format=csv || true
  die "物理卡 $GPU_IDX 空闲显存 ${FREE_GB}GB < ${MIN_FREE_GB}GB —— 改 .env 的 CUDA_VISIBLE_DEVICES（或 --gpu N）换卡，或人工确认后重跑；脚本不会自动停服务"
fi
info "物理卡 $GPU_IDX 空闲显存 ${FREE_GB}GB ✅"

log "预检 · CUDA / 数据产物 / 模板一致性"
"$PY" - "$CONFIG" "$REPO" "$REQUIRE_PROBE" <<'PY' || die "预检未通过（详见上方输出）"
import json, sys
from pathlib import Path
import yaml, torch

cfg_path, repo, require_probe = sys.argv[1], Path(sys.argv[2]), sys.argv[3] == "1"
cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
out = Path(cfg["output"]["dir"])

print(f"  torch {torch.__version__} | cuda={torch.cuda.is_available()} | devices={torch.cuda.device_count()}")
if not torch.cuda.is_available():
    sys.exit("torch 看不到 CUDA")

for name in ("train.parquet", "dev.parquet"):
    p = out / name
    if not p.exists():
        sys.exit(f"缺少 {p}——先跑 python -m prm.build_dataset --config {cfg_path}")
    print(f"  {p}  {p.stat().st_size/1e6:.1f}MB ✅")

man = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
sys.path.insert(0, str(repo))
from prm.prompts import template_hash
if man.get("template_hash") != template_hash():
    sys.exit(f"template_hash 不一致（数据 {man.get('template_hash')} vs 当前 {template_hash()}）"
             "——prm/prompts.py 改过就必须重建数据，否则 prompt 与权重错配")
print(f"  template_hash={template_hash()[:12]}… ✅")
cw = man.get("class_weights", {})
print(f"  类别权重 w_pos={cw.get('w_pos')} w_neg={cw.get('w_neg')} ✅")
counts = man.get("counts") or {}
print(f"  样本数 {counts}")

tr = cfg["train"]
per = int(tr["batch"]["per_device_train_batch_size"]) * int(tr["batch"]["gradient_accumulation_steps"])
n_train = counts.get("train")
if n_train:
    import math
    print(f"  有效 batch={per} → 约 {math.ceil(n_train/per)} steps/epoch × {tr['num_train_epochs']} epochs")
print(f"  max_length={tr['max_length']} lr={tr['lr']} eval_steps={tr['eval_steps']} "
      f"lora={'on' if tr.get('lora') else 'off'} bf16={tr.get('bf16')}")

if require_probe:
    rp = out / "probe_report.json"
    if not rp.exists():
        sys.exit("--require-probe 但缺少 outputs/prm/probe_report.json——先跑 probe")
    rep = json.loads(rp.read_text(encoding="utf-8"))
    print(f"  probe best_pair={rep['best_pair']} auc={rep['pairs'][rep['best_pair']]['auc']:.4f} "
          f"verdict_usable={rep['verdict_usable']}")
    if not rep.get("verdict_usable"):
        sys.exit("probe 门禁未过（best AUC ≤ 0.55）——按 §12 换 verdict_pair 复测后再训练")
PY

log "预检 · run 目录与磁盘"
RUN_DIR="$REPO/outputs/prm/runs/$RUN"
if [ -d "$RUN_DIR" ] && [ -f "$RUN_DIR/run_manifest.json" ] && [ "$RESUME" -eq 0 ] && [ "$FORCE" -eq 0 ]; then
  die "$RUN_DIR 已存在（完成过训练）。要覆盖加 --force，要续训加 --resume"
fi
[ "$RESUME" -eq 1 ] && info "续训检查点: $(ls -d "$RUN_DIR"/checkpoints/checkpoint-* 2>/dev/null | tail -1 || echo '无（将从头训练）')"
AVAIL_GB="$(df -BG --output=avail "$REPO" | tail -1 | tr -dc '0-9')"
[ "${AVAIL_GB:-0}" -ge 10 ] || die "工作区可用磁盘 ${AVAIL_GB}GB < 10GB"
info "可用磁盘 ${AVAIL_GB}GB ✅"

# §1.2 要求：把本次训练用的依赖组合留档（版本漂移已多次咬人，见计划书 §14.3 U1）
FREEZE="$REPO/outputs/prm/requirements-freeze.txt"
if "$PY" -m pip freeze > "$FREEZE" 2>/dev/null; then
  info "依赖快照已刷新: $FREEZE（$(wc -l < "$FREEZE") 项）"
else
  info "⚠️ 依赖快照写入失败（$FREEZE），不阻塞训练"
fi

# ---------------------------------------------------------------- 训练
PY_ARGS=(--config "$CONFIG")
[ -n "$MAX_LENGTH" ] && PY_ARGS+=(--max-length "$MAX_LENGTH")

log "步骤 1/3 · 冒烟训练（20 steps @ 8K）"
if [ "$SKIP_SMOKE" -eq 1 ]; then
  info "已按 --skip-smoke 跳过"
else
  SMOKE_LOG="$LOG_DIR/smoke-${RUN}-${TS}.log"
  info "日志: $SMOKE_LOG"
  if [ "$DRY_RUN" -eq 1 ]; then
    printf '  [dry-run] CUDA_VISIBLE_DEVICES=%s %s -m prm.train_prm %s --run smoke --smoke\n' \
           "${CUDA_VISIBLE_DEVICES:-<未设置>}" "$PY" "${PY_ARGS[*]}"
  else
    "$PY" -m prm.train_prm "${PY_ARGS[@]}" --run smoke --smoke 2>&1 \
      | tee "$SMOKE_LOG" || die "冒烟训练失败，日志见 $SMOKE_LOG"
    grep -q '"best_metric": [0-9]' "$RUN_DIR/../smoke/run_manifest.json" 2>/dev/null \
      && info "冒烟 best_metric 已产出 ✅"
  fi
fi

log "步骤 2/3 · 正式训练（run=$RUN）"
TRAIN_LOG="$LOG_DIR/full-${RUN}-${TS}.log"
FULL_ARGS=("${PY_ARGS[@]}" --run "$RUN")
[ "$RESUME" -eq 1 ] && FULL_ARGS+=(--resume)
info "日志: $TRAIN_LOG（主日志同时汇总在 $MAIN_LOG）"
if [ "$DRY_RUN" -eq 1 ]; then
  printf '  [dry-run] CUDA_VISIBLE_DEVICES=%s %s -m prm.train_prm %s\n' "${CUDA_VISIBLE_DEVICES:-<未设置>}" "$PY" "${FULL_ARGS[*]}"
else
  "$PY" -m prm.train_prm "${FULL_ARGS[@]}" 2>&1 \
    | tee "$TRAIN_LOG" || die "正式训练失败，日志见 $TRAIN_LOG"
fi

log "步骤 3/3 · 评估（§9.2）"
if [ "$NO_EVAL" -eq 1 ]; then
  info "已按 --no-eval 跳过"
elif [ "$DRY_RUN" -eq 1 ]; then
  printf '  [dry-run] %s -m prm.eval_prm %s --run %s --device %s\n' "$PY" "$CONFIG" "$RUN" "$DEVICE"
else
  EVAL_LOG="$LOG_DIR/eval-${RUN}-${TS}.log"
  info "日志: $EVAL_LOG"
  "$PY" -m prm.eval_prm --config "$CONFIG" --run "$RUN" --device "$DEVICE" 2>&1 \
    | tee "$EVAL_LOG" || die "评估失败，日志见 $EVAL_LOG"
fi

log "完成 ✅ run=$RUN"
info "产物目录 : outputs/prm/runs/$RUN/"
info "训练日志 : $TRAIN_LOG"
[ "$NO_EVAL" -eq 0 ] && info "评估产物 : outputs/prm/eval_report.{md,json} + predictions.parquet"
info "主日志   : $MAIN_LOG"
