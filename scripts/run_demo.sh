#!/bin/bash
# ============================================================================
# run_demo.sh —— 确保 phoenix serve 运行 + 以 Phoenix 追踪模式运行 demo
# ----------------------------------------------------------------------------
# 前提：
#   - conda 环境 CodeAgentRL（Agent 运行）与 phoenix（arize-phoenix 服务端）
#     均已就绪（环境搭建见 README §7）；
#   - A800 本地 vllm 已起（8010 端口 gemma-4，无需 API key）。
#
# 行为：
#   1. 若 phoenix serve 未运行：在 phoenix 环境中以 nohup 后台启动（日志
#      /tmp/phoenix_serve.log），并等待 UI（http://localhost:6006）就绪，
#      最长约 180s；超时未就绪则报错退出（不静默继续）。
#   2. 以 --phoenix-tracing 运行 scripts/demo.py：demo.py 假定 serve 已启动
#      （本脚本第 1 步已保证）；运行结束后把执行轨迹导出到
#      outputs/traces/<trace_id>.json。任何一步失败直接抛异常。
#
# 用法：
#   bash scripts/run_demo.sh                     # 默认：带 Phoenix 追踪
#   USE_PHOENIX_TRACING=0 bash scripts/run_demo.sh  # 关闭追踪（不启动 serve）
#
# 服务端会保持运行，下次执行脚本直接复用；停止：pkill -f "phoenix serve"
# ============================================================================

set -euo pipefail

source ~/anaconda3/etc/profile.d/conda.sh

# ---------- 0) 可调配置 ----------
PHOENIX_UI_URL="${PHOENIX_UI_URL:-http://localhost:6006}"
PHOENIX_COLLECTOR_ENDPOINT="${PHOENIX_COLLECTOR_ENDPOINT:-http://localhost:4317}"
PHOENIX_PROJECT="${PHOENIX_PROJECT:-codeagentrl}"
PHOENIX_SERVE_LOG="${PHOENIX_SERVE_LOG:-/tmp/phoenix_serve.log}"
USE_PHOENIX_TRACING="${USE_PHOENIX_TRACING:-1}"

# Agent 运行环境
conda activate CodeAgentRL

# ---------- 1) phoenix serve：未运行则启动并等待就绪 ----------
start_phoenix_serve() {
    echo "[run_demo] 在 phoenix 环境启动 phoenix serve（日志: $PHOENIX_SERVE_LOG）..."
    conda activate phoenix
    # PHOENIX_DISABLE_AGENT_ASSISTANT=true：禁用内置 assistant，避免其启动时的
    # 外部探测在 A800 这类网络不稳 / 高负载主机上拖慢 serve 启动。
    PHOENIX_DISABLE_AGENT_ASSISTANT=true \
        nohup phoenix serve > "$PHOENIX_SERVE_LOG" 2>&1 &
    echo "[run_demo] phoenix serve PID=$!（后台运行）"
    conda activate CodeAgentRL
}

if curl -s -o /dev/null --max-time 3 "$PHOENIX_UI_URL"; then
    echo "[run_demo] phoenix serve 已在运行（$PHOENIX_UI_URL），直接复用"
else
    start_phoenix_serve
    echo "[run_demo] 等待 phoenix serve 就绪（最长约 180s；A800 负载高时启动较慢）..."
    ready=0
    for i in $(seq 1 36); do
        if curl -s -o /dev/null --max-time 3 "$PHOENIX_UI_URL"; then
            echo "[run_demo] phoenix serve 就绪（约 $((i * 5))s）"
            ready=1
            break
        fi
        sleep 5
    done
    if [ "$ready" -ne 1 ]; then
        echo "[run_demo] 错误：phoenix serve 未在 180s 内就绪，请检查日志 $PHOENIX_SERVE_LOG" >&2
        exit 1
    fi
fi

# ---------- 2) 运行 demo（默认开启 Phoenix 追踪） ----------
# 注意：
# - 基础镜像来自项目根 .env（CODEAGENTRL_IMAGE=codeagentrl-agent:ubuntu24），
#   demo.py 会在 import agent 之前 load_dotenv（必须先于 import 才生效）。
# - 模型名必须带 provider 前缀：litellm 需要 openai/gemma-4 才能路由到
#   OpenAI 兼容端点（裸 gemma-4 会报 "LLM Provider NOT provided"）。
# - --phoenix-tracing：demo.py 假定 serve 已启动（本脚本第 1 步已保证）；
#   导出轨迹等任何一步失败都会直接抛异常。

TRACE_ARGS=()
if [ "$USE_PHOENIX_TRACING" = "1" ]; then
    TRACE_ARGS+=(--phoenix-tracing)
fi

CODEAGENTRL_REPO_CACHE_DIR=/tmp/demo_repo_cache \
PHOENIX_COLLECTOR_ENDPOINT="$PHOENIX_COLLECTOR_ENDPOINT" \
PHOENIX_PROJECT="$PHOENIX_PROJECT" \
PHOENIX_REST_URL="$PHOENIX_UI_URL" \
python scripts/demo.py hasaki1025/MyRpc \
    --model openai/gemma-4 --base-url http://localhost:8010/v1 \
    --keep-container "${TRACE_ARGS[@]}"
