#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause
#
# 迁移后自检（也可迁移前当"清单核对"跑）。只读，不改任何文件。
#   bash scripts/check_migration.sh            # 检查当前仓库
#   REPO=/path/to/CodeAgentRL bash scripts/check_migration.sh
#
# 退出码 = 必选项失败数（0 = 必选项全通过；可选项失败不影响退出码）。
set -uo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO" || { echo "找不到仓库目录 $REPO"; exit 1; }

PY="${PRM_PY:-$REPO/.venv-prm/bin/python}"
BASE_MODEL="${PRM_BASE_MODEL:-/media/shared_e/models/Qwen3.5-4B}"
FAIL=0
WARN=0

ok()   { printf '  \033[32m✅\033[0m %s\n' "$1"; }
bad()  { printf '  \033[31m❌\033[0m %s\n' "$1"; FAIL=$((FAIL + 1)); }
warn() { printf '  \033[33m⚠️\033[0m  %s\n' "$1"; WARN=$((WARN + 1)); }

hr() { printf '\n\033[1m== %s ==\033[0m\n' "$1"; }

sz() {  # 人类可读体积
  [ -e "$1" ] || { echo "-"; return; }
  du -sh "$1" 2>/dev/null | awk '{print $1}'
}

hr "1. 代码（应来自 git clone）"
if git rev-parse --git-dir >/dev/null 2>&1; then
  ok "git 仓库存在，HEAD=$(git rev-parse --short HEAD) 分支=$(git rev-parse --abbrev-ref HEAD)"
else
  bad "不是 git 仓库（应该 git clone 而来）"
fi
for f in prm/train_prm.py prm/data.py prm/modeling.py prm/eval_prm.py config/prm.yaml \
         scripts/train_prm_m4.sh scripts/run_prm_m4_gpu.sh scripts/bench_prm_step.py \
         docs/prm_training_plan.md docs/prm_m4_runbook.md; do
  [ -f "$f" ] && ok "存在 $f" || bad "缺失 $f"
done

hr "2. Python 环境（建议在新机重建，勿直接拷 venv）"
if [ -x "$PY" ]; then
  ok "解释器 $PY（$("$PY" -V 2>&1)）"
  "$PY" - <<'PYEOF' && ok "torch/transformers/peft/fla 可导入" || bad "依赖导入失败（见上方报错）"
import importlib, sys
mods = ["torch", "transformers", "peft", "fla", "pyarrow", "sklearn"]
missing = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception as e:
        missing.append(f"{m}({type(e).__name__})")
import torch
print(f"     torch={torch.__version__} cuda_available={torch.cuda.is_available()} "
      f"devices={torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"     gpu={torch.cuda.get_device_name(0)}")
sys.exit(1 if missing else 0)
PYEOF
  "$PY" -c "import flash_attn; print('     flash_attn', flash_attn.__version__)" 2>/dev/null \
    && ok "flash_attn 可用（训练应看到 attn=flash_attention_2）" \
    || warn "flash_attn 不可用 → 训练会自动降级 sdpa（不报错，但更慢）"
else
  bad "缺少解释器 $PY（重建：python3.12 -m venv .venv-prm && 装 requirements-freeze.txt；或用 PRM_PY 指定）"
fi

hr "3. 基座模型（仓库外，必须手工迁移或改 config 路径）"
if [ -d "$BASE_MODEL" ]; then
  ok "存在 $BASE_MODEL（$(sz "$BASE_MODEL")）"
  ls "$BASE_MODEL"/*.safetensors >/dev/null 2>&1 \
    && ok "含 safetensors 权重" || bad "$BASE_MODEL 下没有 *.safetensors"
else
  bad "缺少基座模型 $BASE_MODEL —— 迁移该目录，或改 config/prm.yaml 的 tokenizer 与 train.base_model"
fi

hr "4. PRM 训练/评估数据（gitignored，必须手工迁移）"
for f in outputs/prm/train.parquet outputs/prm/dev.parquet outputs/prm/test.parquet \
         outputs/prm/manifest.json; do
  [ -f "$f" ] && ok "$f（$(sz "$f")）" || bad "缺失 $f"
done
if [ -f outputs/prm/runs/m4-v1/model.safetensors ]; then
  ok "已训练产物 outputs/prm/runs/m4-v1/（$(sz outputs/prm/runs/m4-v1)）"
  [ -d outputs/prm/runs/m4-v1/checkpoints ] \
    && ok "含 checkpoints（可续训）" || warn "无 checkpoints（只能评估，不能续训）"
else
  bad "缺少 outputs/prm/runs/m4-v1/model.safetensors（评估/下游都要用）"
fi

hr "5. 数据重建链（只评估/训练可不迁；要重建数据则必需）"
for f in outputs/batch500/state.db outputs/mcts/splits.parquet; do
  [ -f "$f" ] && ok "$f（$(sz "$f")）" || warn "缺少 $f → 无法用 build_dataset 重建数据"
done
[ -f outputs/batch500/state.db-wal ] && \
  { [ -s outputs/batch500/state.db-wal ] \
      && warn "state.db-wal 非空，迁移时必须与 state.db / state.db-shm 一起拷贝" \
      || ok "state.db-wal 为空（干净状态）"; }
for f in outputs/prm/oversize_skip_train.json outputs/prm/oversize_skip_dev.json; do
  [ -f "$f" ] && ok "缓存 $f" || warn "缺 $f → 首次训练会重扫（约 4 分钟，自动生成）"
done

hr "6. 运行时配置与凭据"
[ -f .env ] && ok ".env 存在" || bad "缺少 .env（从 .env_template 复制并填值：CUDA_VISIBLE_DEVICES / CODEAGENTRL_IMAGE / PRM_PY 等）"
grep -q '^CUDA_VISIBLE_DEVICES=' .env 2>/dev/null \
  && ok "已设置 CUDA_VISIBLE_DEVICES=$(grep '^CUDA_VISIBLE_DEVICES=' .env | cut -d= -f2)" \
  || warn ".env 未设置 CUDA_VISIBLE_DEVICES（默认用物理卡 0）"
[ -f .env ] && grep -q '^PRM_PY=' .env && ok ".env 指定了 PRM_PY" \
  || warn ".env 未指定 PRM_PY → 脚本按 .venv-prm/bin/python 兜底"
if git remote -v 2>/dev/null | grep -q 'git@'; then
  if timeout 15 git ls-remote --heads origin >/dev/null 2>&1; then
    ok "GitHub 远端可达（SSH）"
  else
    warn "GitHub 远端不可达：新机需要 ~/.ssh/config 的 Host 别名与私钥，或改用 HTTPS 远端"
  fi
fi
ls "$REPO"/flash_attn-*.whl >/dev/null 2>&1 \
  && ok "本地 flash-attn 轮存在（重建 venv 用）" \
  || warn "无本地 flash-attn 轮 → 重建 venv 时需另找匹配 ABI 的轮子"

hr "7. 仅 Agent/Rollout 侧需要（本阶段 PRM 训练不需要）"
state=$( { command -v docker >/dev/null 2>&1 && timeout 20 docker images codeagentrl-agent --format '{{.Repository}}:{{.Tag}}' 2>/dev/null; } || true )
if [ -n "$state" ]; then
  ok "docker 镜像：$(echo "$state" | tr '\n' ' ')"
else
  warn "未发现 codeagentrl-agent 镜像（rollout 需要：docker load 或按 scripts/Dockerfile 重建）"
fi
[ -d "${CODEAGENTRL_REPO_CACHE_DIR:-/media/shared_e/lyq/repos}" ] \
  && ok "repo 缓存 ${CODEAGENTRL_REPO_CACHE_DIR:-/media/shared_e/lyq/repos}（$(sz "${CODEAGENTRL_REPO_CACHE_DIR:-/media/shared_e/lyq/repos}")）" \
  || warn "无 repo 缓存（rollout 会重新拉仓库）"

printf '\n\033[1m== 结论 ==\033[0m\n'
if [ "$FAIL" -eq 0 ]; then
  printf '  必选项全部通过 ✅（可选项警告 %d 条）\n' "$WARN"
  printf '  下一步：./scripts/train_prm_m4.sh --dry-run  然后  ./scripts/run_prm_m4_gpu.sh smoke\n'
else
  printf '  \033[31m必选项失败 %d 条\033[0m，可选项警告 %d 条 —— 按上面 ❌ 逐条补齐\n' "$FAIL" "$WARN"
fi
exit "$FAIL"
