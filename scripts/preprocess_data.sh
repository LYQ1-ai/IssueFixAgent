#!/bin/bash
# ============================================================================
# preprocess_data.sh —— 端到端数据预处理：原始 swe_smith parquet → 最终数据集
# ----------------------------------------------------------------------------
# 串起 M0 数据预处理的全部步骤（每步幂等、可单独重跑）：
#
#   1. 解析 + 过滤 + GT 抽取   python -m mcts.instances build
#        → outputs/mcts/instances.parquet + data_report.{json,md}
#        （从 data/swe_smith/{train,validation}.parquet 读原始数据（可用
#          --data-dir 指定其它位置），剔除非法/超长实例，抽三粒度 gold）
#   2. 三层划分               python -m mcts.splits build
#        → outputs/mcts/splits.parquet + splits_report.{json,md}
#        （60% repo 生成池 + PRM train/dev/test 80/10/10，同 repo 不跨 split）
#   3. （可选）仓库缓存       python scripts/batch_repo_pull.py
#        → 把数据集全部仓库批量拉取到本地 zip 缓存（rollout 网络预热）；
#          仅当显式传 --cache-dir（或用环境变量 CODEAGENTRL_REPO_CACHE_DIR）
#          时执行，默认跳过。
#
# 用法（项目根目录或任意位置，需 conda 环境 CodeAgentRL）：
#   bash scripts/preprocess_data.sh                              # 步骤 1+2（默认数据 data/swe_smith）
#   bash scripts/preprocess_data.sh --cache-dir /data/repo_cache # 1+2+3（推荐）
#   bash scripts/preprocess_data.sh --data-dir ref_papers/codescout/data/swe_smith
#   bash scripts/preprocess_data.sh --skip-splits --cache-dir /data/repo_cache
#   bash scripts/preprocess_data.sh --cache-limit 5              # 调试：只拉前 5 个仓库
#   bash scripts/preprocess_data.sh --dry-run                    # 只打印执行计划
#   bash scripts/preprocess_data.sh --cache-ignore-failures      # 缓存失败不阻断（仅警告）
#
# 退出码：任意非可选步骤失败或（默认）缓存步骤失败 → 1；全部成功 → 0。
# 缓存步骤失败时直接重跑本脚本即可续拉（batch_repo_pull 已成功的仓库自动跳过）。
# ============================================================================

set -euo pipefail

source ~/anaconda3/etc/profile.d/conda.sh
conda activate CodeAgentRL

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

PYTHON="${PYTHON:-python}"
DATA_DIR="${DATA_DIR:-data/swe_smith}"   # 原始数据目录（--data-dir 可覆盖，默认 data/swe_smith）
OUTPUT_DIR="outputs/mcts"

# ---------- 命令行参数 ----------
CACHE_DIR="${CODEAGENTRL_REPO_CACHE_DIR:-}"   # 默认：不启用缓存步骤
DO_INSTANCES=1
DO_SPLITS=1
CACHE_WORKERS=4
CACHE_RETRIES=2
CACHE_LIMIT=""
CACHE_IGNORE_FAILURES=0
DRY_RUN=0

usage() {
    sed -n '2,50p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
}

while [ $# -gt 0 ]; do
    case "$1" in
        --data-dir)            DATA_DIR="$2"; shift 2 ;;
        --cache-dir)           CACHE_DIR="$2"; shift 2 ;;
        --skip-instances)      DO_INSTANCES=0; shift ;;
        --skip-splits)         DO_SPLITS=0; shift ;;
        --cache-workers)       CACHE_WORKERS="$2"; shift 2 ;;
        --cache-retries)       CACHE_RETRIES="$2"; shift 2 ;;
        --cache-limit)         CACHE_LIMIT="$2"; shift 2 ;;
        --cache-ignore-failures) CACHE_IGNORE_FAILURES=1; shift ;;
        --dry-run)             DRY_RUN=1; shift ;;
        -h|--help)             usage ;;
        *) echo "未知参数: $1" >&2; usage ;;
    esac
done

# ---------- 0) 预检 ----------
echo "[preprocess] 项目根目录: $PROJECT_ROOT"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "[preprocess] 错误：找不到 python（$PYTHON），请先激活 conda 环境 CodeAgentRL" >&2
    exit 1
fi
echo "[preprocess] Python: $($PYTHON --version 2>&1)"

if [ ! -f "$DATA_DIR/train.parquet" ]; then
    echo "[preprocess] 错误：找不到原始数据 $DATA_DIR/train.parquet（默认 data/swe_smith，可用 --data-dir 或环境变量 DATA_DIR 指定）" >&2
    exit 1
fi

run_step() {
    local name="$1"; shift
    if [ "$DRY_RUN" = "1" ]; then
        echo "[preprocess] [dry-run] 将执行 $name: $*"
        return 0
    fi
    local t0=$SECONDS
    echo "[preprocess] ===== $name 开始 ====="
    if "$@"; then
        echo "[preprocess] ===== $name 完成（$((SECONDS - t0))s）====="
    else
        local rc=$?
        echo "[preprocess] ===== $name 失败（退出码 $rc，耗时 $((SECONDS - t0))s）=====" >&2
        return "$rc"
    fi
}

# ---------- 1) 解析 + 过滤 + GT 抽取 ----------
if [ "$DO_INSTANCES" = "1" ]; then
    run_step "解析+过滤+GT抽取 (mcts.instances)" \
        "$PYTHON" -m mcts.instances build --data-dir "$DATA_DIR" --output "$OUTPUT_DIR"
else
    echo "[preprocess] 跳过 mcts.instances（--skip-instances）"
fi

# ---------- 2) 三层划分 ----------
if [ "$DO_SPLITS" = "1" ]; then
    if [ ! -f "$OUTPUT_DIR/instances.parquet" ]; then
        echo "[preprocess] 错误：$OUTPUT_DIR/instances.parquet 不存在，无法划分（先跑步骤 1）" >&2
        exit 1
    fi
    run_step "三层划分 (mcts.splits)" \
        "$PYTHON" -m mcts.splits build --instances "$OUTPUT_DIR/instances.parquet" --output "$OUTPUT_DIR"
else
    echo "[preprocess] 跳过 mcts.splits（--skip-splits）"
fi

# ---------- 3) （可选）仓库缓存 ----------
if [ -n "$CACHE_DIR" ]; then
    CACHE_ARGS=(--cache-dir "$CACHE_DIR" --workers "$CACHE_WORKERS" --retries "$CACHE_RETRIES")
    [ -n "$CACHE_LIMIT" ] && CACHE_ARGS+=(--limit "$CACHE_LIMIT")
    # 优先用实例表里的仓库（与训练一致）；实例表缺失则退化为读原始 parquet
    if [ -f "$OUTPUT_DIR/instances.parquet" ]; then
        CACHE_ARGS+=(--instances "$OUTPUT_DIR/instances.parquet")
    else
        CACHE_ARGS+=(--data-dir "$DATA_DIR")
    fi
    if run_step "仓库缓存 (batch_repo_pull)" "$PYTHON" scripts/batch_repo_pull.py "${CACHE_ARGS[@]}"; then
        :
    else
        rc=$?
        if [ "$CACHE_IGNORE_FAILURES" = "1" ]; then
            echo "[preprocess] 警告：仓库缓存有失败项（退出码 $rc），但 --cache-ignore-failures 已设置，继续" >&2
        else
            echo "[preprocess] 错误：仓库缓存未全部成功（退出码 $rc）；可原样重跑本脚本续拉" >&2
            exit 1
        fi
    fi
else
    echo "[preprocess] 跳过仓库缓存（未指定 --cache-dir，也未设置 CODEAGENTRL_REPO_CACHE_DIR）"
    echo "[preprocess] 提示：训练前建议启用缓存，如 --cache-dir /data/repo_cache"
fi

# ---------- 汇总 ----------
echo "=============================================================="
echo "[preprocess] 数据预处理完成。产物："
ls -lh "$OUTPUT_DIR" 2>/dev/null | awk 'NR>1 && $9 != "." {print "  " $9 " (" $5 ")"}' || true
if [ -n "$CACHE_DIR" ]; then
    nzips=$(ls "$CACHE_DIR"/*.zip 2>/dev/null | wc -l || true)
    echo "[preprocess] 仓库缓存目录: $CACHE_DIR（zip 数: $nzips）"
fi
echo "[preprocess] 下一步：阶段 1 M1 单 rollout（PLAN §2.2/§5）"
