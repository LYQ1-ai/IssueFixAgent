#!/bin/bash
# ============================================================================
# run_sglang_qwen4b.sh —— 快捷启动 sglang Qwen3.5-4B（本地 OpenAI 兼容服务）
# 用法：bash scripts/run_sglang_qwen4b.sh
# GPU1 :30000（GPU0 已被 vLLM gemma-4 占用 :8010）
# 就绪后对接：bash scripts/run_rollout.sh --trees 10 --seed 42 \
#     --model openai/qwen3.5-4B --base-url http://localhost:30000/v1
# ============================================================================

source ~/anaconda3/etc/profile.d/conda.sh
conda activate sglang

CUDA_VISIBLE_DEVICES=1 python -m sglang.launch_server \
    --served-model-name qwen3.5-4B \
    --model-path /media/shared_e/models/Qwen3.5-4B \
    --host 0.0.0.0 \
    --port 30000 \
    --context-length 65536 \
    --max-running-requests 40 \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    --chunked-prefill-size 8192 \
    --kv-cache-dtype auto \
    --mem-fraction-static 0.90 \
    --trust-remote-code

# 说明：
# - --kv-cache-dtype auto：A800（Ampere sm_80）不支持 fp8 KV cache，用默认 bf16；
# - 若启动时 CUDA graph 捕获崩溃（A800 上偶发），在命令末尾追加 --disable-cuda-graph；
# - 健康检查：curl http://localhost:30000/v1/models
