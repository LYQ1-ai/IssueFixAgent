#!/bin/bash
# ============================================================================
# run_rollout.sh —— PRM 训练数据 MCTS Rollout 完整脚本（阶段 1 数据生成）
# ----------------------------------------------------------------------------
# 端到端串起"读取预处理数据集 → 按种子抽样 → 真实 MCTS Rollout → SQLite 落库
# + 数据报告"的完整流程（每实例 = 一棵 MCTS 树）：
#
#   0. 数据检查      instances.parquet / splits.parquet 缺失时自动调用
#                    scripts/preprocess_data.sh（M0 预处理，幂等可重跑）
#   1. 抽样          python -m mcts.run_mcts --sample <trees> --seed <seed>
#                    （从生成池按种子采样 trees 个实例，每个实例一棵树）
#   2. MCTS Rollout  真实执行：每实例 root rollout × N → MC 门控 → select/locate
#                    二分（probe 节点 N 次 rollout 并发）→ best/leaf/add 标注；
#                    容器创建即用、用完即销毁；结果常驻内存 + 定期/完成即写 SQLite
#   3. 报告          run_mcts 自动生成 outputs/mcts/rollout_report.{json,md}
#                    （吞吐/失败率/回报/标注统计，PLAN §5 M3 验收口径）
#
# 用法（项目根目录，需 conda 环境 CodeAgentRL；本地 vLLM 已起 gemma-4）：
#   bash scripts/run_rollout.sh --trees 10 --seed 42
#   bash scripts/run_rollout.sh --trees 100 --seed 42 --concurrency 40 --create-concurrency 8
#   bash scripts/run_rollout.sh --trees 20 --seed 7 --resume --max-rollouts 200   # 预算熔断
#   bash scripts/run_rollout.sh --trees 5  --seed 1 --dry-run                     # 管道自检（不碰 Docker/LLM）
#   bash scripts/run_rollout.sh --report-only --trees 10 --seed 42                # 只生成报告（不跑 rollout）
#
# 退出码：任意步骤失败 → 1；全部成功 → 0。中断后可原样重跑（--resume 续跑）。
# ============================================================================

set -euo pipefail

source ~/anaconda3/etc/profile.d/conda.sh
conda activate CodeAgentRL

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

PYTHON="${PYTHON:-python}"
OUTPUT_DIR="outputs/mcts"

# ---------- 命令行参数 ----------
SEED=42
TREES=""                       # 采样实例数 = MCTS 树数（必填）
CONCURRENCY=8                  # 全局 rollout 并发（A800 目标 40–50）
CREATE_CONCURRENCY=8           # 容器并发创建数（只限同时创建数，不限总次数）
N_ROLLOUTS=5                   # 每节点 rollout 次数 N
MAX_ITERATIONS=20              # 标注循环轮数上限
STEP_LIMIT=20                  # agent 回合上限（2026-08-27 起 8→20）
REWARD_THRESHOLD=0.5           # correct 阈值 θ（结构化定位 F1 ∈ [0,3]）
MAGIC_SUBMIT=""                # 空 = 按 mcts/config.yaml submit.magic_submit
MAX_ROLLOUTS=0                 # 全局 rollout 预算（0 = 不限）
MAX_ROLLOUTS_PER_INSTANCE=0
MODEL="openai/gemma-4"
BASE_URL="http://localhost:8010/v1"
API_KEY=""
RESUME=0
DRY_RUN=0
REPORT_ONLY=0
SKIP_PREPROCESS=0
DATA_DIR="data/swe_smith"      # 自动 preprocess 时的原始数据目录
PIPELINE_ARGS=()

usage() {
    sed -n '2,35p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
}

while [ $# -gt 0 ]; do
    case "$1" in
        --seed)                  SEED="$2"; shift 2 ;;
        --trees)                 TREES="$2"; shift 2 ;;
        --concurrency)           CONCURRENCY="$2"; shift 2 ;;
        --create-concurrency)    CREATE_CONCURRENCY="$2"; shift 2 ;;
        --n-rollouts)            N_ROLLOUTS="$2"; shift 2 ;;
        --max-iterations)        MAX_ITERATIONS="$2"; shift 2 ;;
        --step-limit)            STEP_LIMIT="$2"; shift 2 ;;
        --reward-threshold)      REWARD_THRESHOLD="$2"; shift 2 ;;
        --magic-submit)          MAGIC_SUBMIT=1; shift ;;
        --no-magic-submit)       MAGIC_SUBMIT=0; shift ;;
        --max-rollouts)          MAX_ROLLOUTS="$2"; shift 2 ;;
        --max-rollouts-per-instance) MAX_ROLLOUTS_PER_INSTANCE="$2"; shift 2 ;;
        --model)                 MODEL="$2"; shift 2 ;;
        --base-url)              BASE_URL="$2"; shift 2 ;;
        --api-key)               API_KEY="$2"; shift 2 ;;
        --resume)                RESUME=1; shift ;;
        --no-resume)             RESUME=0; shift ;;
        --dry-run)               DRY_RUN=1; shift ;;
        --report-only)           REPORT_ONLY=1; shift ;;
        --skip-preprocess)       SKIP_PREPROCESS=1; shift ;;
        --data-dir)              DATA_DIR="$2"; shift 2 ;;
        --output)                OUTPUT_DIR="$2"; shift 2 ;;
        -h|--help)               usage ;;
        *) echo "未知参数: $1" >&2; usage ;;
    esac
done

if [ -z "$TREES" ]; then
    echo "[rollout] 错误：必须指定 --trees N（采样实例数 = MCTS 树数）" >&2
    usage
fi

# ---------- 0) 预检 + 数据检查（缺失时自动预处理） ----------
echo "[rollout] 项目根目录: $PROJECT_ROOT"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "[rollout] 错误：找不到 python（$PYTHON），请先激活 conda 环境 CodeAgentRL" >&2
    exit 1
fi
echo "[rollout] Python: $($PYTHON --version 2>&1)"
echo "[rollout] 采样: seed=$SEED trees=$TREES 并发=$CONCURRENCY 创建并发=$CREATE_CONCURRENCY N=$N_ROLLOUTS"

if [ ! -f "$OUTPUT_DIR/instances.parquet" ] || [ ! -f "$OUTPUT_DIR/splits.parquet" ]; then
    if [ "$SKIP_PREPROCESS" = "1" ]; then
        echo "[rollout] 错误：$OUTPUT_DIR/instances.parquet 或 splits.parquet 缺失（--skip-preprocess 已设置，不自动预处理）" >&2
        exit 1
    fi
    echo "[rollout] 数据缺失，自动执行数据预处理（M0）：bash scripts/preprocess_data.sh --data-dir $DATA_DIR"
    if [ "$DRY_RUN" = "1" ]; then
        echo "[rollout] [dry-run] 将执行: bash scripts/preprocess_data.sh --data-dir $DATA_DIR"
    else
        bash scripts/preprocess_data.sh --data-dir "$DATA_DIR"
    fi
else
    echo "[rollout] 数据就绪: $OUTPUT_DIR/{instances,splits}.parquet"
fi

# ---------- 1+2+3) 抽样 + MCTS Rollout + 报告 ----------
PIPELINE_ARGS=(
    --sample "$TREES" --seed "$SEED"
    --concurrency "$CONCURRENCY"
    --create-concurrency "$CREATE_CONCURRENCY"
    --n-rollouts "$N_ROLLOUTS"
    --max-iterations "$MAX_ITERATIONS"
    --step-limit "$STEP_LIMIT"
    --reward-threshold "$REWARD_THRESHOLD"
    --model "$MODEL" --base-url "$BASE_URL"
    --output "$OUTPUT_DIR"
)
[ -n "$API_KEY" ] && PIPELINE_ARGS+=(--api-key "$API_KEY")
[ "$MAX_ROLLOUTS" != "0" ] && PIPELINE_ARGS+=(--max-rollouts "$MAX_ROLLOUTS")
[ "$MAX_ROLLOUTS_PER_INSTANCE" != "0" ] && PIPELINE_ARGS+=(--max-rollouts-per-instance "$MAX_ROLLOUTS_PER_INSTANCE")
[ "$RESUME" = "1" ] && PIPELINE_ARGS+=(--resume)
[ "$DRY_RUN" = "1" ] && PIPELINE_ARGS+=(--dry-run)
[ "$REPORT_ONLY" = "1" ] && PIPELINE_ARGS+=(--report-only)
[ -n "$MAGIC_SUBMIT" ] && {
    [ "$MAGIC_SUBMIT" = "1" ] && PIPELINE_ARGS+=(--magic-submit)
    [ "$MAGIC_SUBMIT" = "0" ] && PIPELINE_ARGS+=(--no-magic-submit)
}

if [ "$DRY_RUN" = "1" ] || [ "$REPORT_ONLY" = "1" ]; then
    # dry-run / report-only 不读原始数据，跳过存在性检查的误导信息
    :
fi

echo "[rollout] ===== 开始 MCTS Rollout（$TREES 棵树，seed=$SEED）====="
t0=$SECONDS
"$PYTHON" -m mcts.run_mcts "${PIPELINE_ARGS[@]}"
rc=$?
echo "[rollout] ===== MCTS Rollout 结束（$((SECONDS - t0))s，退出码 $rc）====="
[ "$rc" -ne 0 ] && exit "$rc"

# ---------- 汇总 ----------
echo "=============================================================="
echo "[rollout] 产物："
ls -lh "$OUTPUT_DIR"/rollout_report.* 2>/dev/null | awk 'NR>1 {print "  " $9 " (" $5 ")"}' || true
ls -lh "$OUTPUT_DIR/state.db" 2>/dev/null | awk 'NR>1 {print "  " $9 " (" $5 ")"}' || true
echo "[rollout] 下一步：PRM 训练数据构建（PLAN §3.1，从 state.db 的 rollouts/annotations 展开步级样本）"
