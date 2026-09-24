# SPDX-License-Identifier: BSD-3-Clause

"""prm/train_prm.py 离线冒烟（torch 门控）：tiny 模型过完整 HF Trainer 循环。

验证训练机制本身（compute_loss / prediction_step / dev_auc 选优 / 产物齐全 /
run_manifest 字段），不依赖 GPU 与真实基座。风险点 = transformers 版本 API
差异，此测试在 PRM 环境最先暴露。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))
sys.path.insert(0, str(TESTS_DIR.parent))

pytest.importorskip("torch")
pytest.importorskip("transformers")

import pyarrow as pa
import pyarrow.parquet as pq  # noqa: E402

from prm.build_dataset import SCHEMA  # noqa: E402
from prm.prompts import template_hash  # noqa: E402
from prm.train_prm import train  # noqa: E402
from prm_torch_fixtures import (  # noqa: E402
    build_tiny_tokenizer,
    make_long_output,
    make_tiny_messages,
)

CORRECT_ID, INCORRECT_ID = 2, 3  # tiny tokenizer 的 verdict token ids


def _sample_row(i: int, split: str, label_binary: int, token_len: int) -> dict:
    msgs = make_tiny_messages(1)
    for m in msgs:  # parquet 列 arguments = canonical dict 的 JSON 序列化
        for tc in m.get("tool_calls") or []:
            if isinstance(tc.get("function", {}).get("arguments"), dict):
                tc["function"]["arguments"] = json.dumps(tc["function"]["arguments"])
    return {
        "sample_id": f"{split}-{i:04d}",
        "instance_id": f"inst-{i % 4:02d}",
        "repo": "repo",
        "split": split,
        "node_key": f"key-{i:04d}",
        "step_index": (i % 3) + 1,
        "step_count": 3,
        "messages": msgs,
        "label": float(label_binary),
        "label_binary": label_binary,
        "label_soft": float(label_binary),
        "label_source": "node_mc",
        "mc_score": float(label_binary),
        "n_rollouts": 5,
        "visits": 5,
        "in_pool": 1,
        "rendered_tokens": token_len,
    }


@pytest.fixture(scope="module")
def tiny_world(tmp_path_factory):
    """tiny 模型目录 + train/dev parquet + 训练 config。"""
    from prm.modeling import VerdictScorer
    from prm_torch_fixtures import build_tiny_model

    tmp = tmp_path_factory.mktemp("prm_train_smoke")
    tok = build_tiny_tokenizer()
    model_dir = tmp / "tiny-model"
    scorer = VerdictScorer(build_tiny_model(vocab_size=64), (CORRECT_ID, INCORRECT_ID))
    model_dir.mkdir(parents=True)
    scorer.save_pretrained(str(model_dir))   # → model.safetensors + config.json
    tok.save_pretrained(str(model_dir))

    def write_parquet(name: str, rows: list[dict]) -> Path:
        path = tmp / name
        pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), path)
        return path

    train_rows = [_sample_row(i, "train", i % 2, 800) for i in range(8)]
    dev_rows = [_sample_row(i, "dev", i % 2, 800) for i in range(4)]
    # §7.2 边界样本：末步 assistant 内容超预算 → 必须被跳过而不是崩掉训练
    # （2026-09-15 真机：19,862 条里 1 条，在 DataLoader worker 抛错，白跑 10 h）
    bad = _sample_row(99, "train", 1, 100000)
    bad["sample_id"] = "train-bad"
    next(m for m in bad["messages"] if m["role"] == "assistant")["content"] = \
        make_long_output(800)
    train_rows.append(bad)
    write_parquet("train.parquet", train_rows)
    write_parquet("dev.parquet", dev_rows)

    cfg = {
        "output": {"dir": str(tmp)},
        "train": {
            "base_model": str(model_dir),
            "max_length": 512,
            "verdict_pair": ["Correct", "Incorrect"],
            "lora": None,
            "batch": {"per_device_train_batch_size": 1, "gradient_accumulation_steps": 2},
            "lr": 1e-3,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.05,
            "num_train_epochs": 1,
            "bf16": False,
            "gradient_checkpointing": False,
            "attn_implementation": "sdpa",
            "seed": 42,
            "eval_steps": 1,
            "save_steps": 1,
            "save_total_limit": 2,
            "metric_for_best_model": "dev_auc",
            "greater_is_better": True,
            "early_stopping_patience": 3,
            "smoke_max_steps": 4,
        },
    }
    return {"tmp": tmp, "cfg": cfg, "tokenizer": tok, "model_dir": model_dir}


class TestTrainSmoke:
    def test_offline_training_roundtrip(self, tiny_world):
        run_dir = train(tiny_world["cfg"], "smoketest", smoke=True)
        assert (run_dir / "model.safetensors").exists()
        assert (run_dir / "run_manifest.json").exists()
        assert (run_dir / "truncation_stats.json").exists()
        assert (run_dir / "tokenizer_config.json").exists() or (
            run_dir / "tokenizer.json").exists()
        # U2：有效配置快照（run 目录自包含）
        assert (run_dir / "config.yaml").exists()
        snap = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
        assert snap["train"]["max_length"] == 512          # 冒烟 fixture 的定稿值
        assert snap["train"]["batch"]["per_device_train_batch_size"] == 1
        assert snap["run"]["name"] == "smoketest" and snap["run"]["smoke"] is True

        manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        assert manifest["verdict_pair"] == ["Correct", "Incorrect"]
        assert manifest["verdict_token_ids"] == [CORRECT_ID, INCORRECT_ID]
        assert manifest["template_hash"] == template_hash()
        assert manifest["n_train"] == 8 and manifest["n_dev"] == 4
        assert manifest["n_dev_eval"] == 4                 # dev 只有 4 条，冒烟上限 64 不裁
        assert manifest["eval_max_samples"] == 64          # 冒烟默认上限（smoke_eval_max_samples）
        assert manifest["peak_gpu_memory"] is None          # U7：CPU 下为 None
        assert manifest["best_metric"] is not None      # dev 评估实际跑过
        assert 0.0 <= manifest["best_metric"] <= 1.0
        assert manifest["env"]["torch"]
        # 内核化/attn 实际值要留档（吞吐归因用，§14.6）
        assert manifest["attn_implementation"] == "sdpa"
        assert manifest["attn_implementation_effective"] == "sdpa"
        assert manifest["use_kernels"] is False          # CPU/无 kernels 包 → 参考实现
        assert manifest["gpu"] is None
        assert manifest["steps_per_epoch"] == 4          # ceil(8 / (1×2))，9 条里 1 条被剔除
        assert manifest["n_train"] == 8
        assert manifest["oversize_skipped"]["n_train"] == 1
        assert manifest["oversize_skipped"]["train_sample_ids"] == ["train-bad"]
        cfg_snap = yaml.safe_load(
            (run_dir / "config.yaml").read_text(encoding="utf-8"))
        assert cfg_snap["run"]["use_kernels"] is False
        assert cfg_snap["run"]["eval_max_samples_effective"] == 64

    def test_eval_max_samples_subsets_dev(self, tiny_world):
        """U9：eval_max_samples 只评前 N 条（bs=1 下控制评估成本的开关）。"""
        import copy

        cfg = copy.deepcopy(tiny_world["cfg"])
        cfg["train"]["eval_max_samples"] = 2
        run_dir = train(cfg, "smoketest_evalsub", smoke=True)
        manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        assert manifest["n_dev"] == 4          # 全量条数仍如实记录
        assert manifest["n_dev_eval"] == 2     # 实际评估条数
        assert manifest["best_metric"] is not None

    def test_smoke_caps_eval_cost(self, tiny_world):
        """冒烟默认把 dev 评估裁到 smoke_eval_max_samples（否则钱全花在评估上）。"""
        import copy

        cfg = copy.deepcopy(tiny_world["cfg"])
        cfg["train"]["smoke_eval_max_samples"] = 3
        run_dir = train(cfg, "smoketest_smokecap", smoke=True)
        manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        assert manifest["eval_max_samples"] == 3
        assert manifest["n_dev_eval"] == 3

    def test_cli_eval_max_samples_overrides_smoke_cap(self, tiny_world):
        """--eval-max-samples 优先级最高（含冒烟），<=0 视为全量。"""
        import copy

        cfg = copy.deepcopy(tiny_world["cfg"])
        cfg["train"]["smoke_eval_max_samples"] = 3
        run_dir = train(cfg, "smoketest_evalcli", smoke=True, eval_max_samples=2)
        manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        assert manifest["eval_max_samples"] == 2

    def test_resolve_eval_max_priority(self):
        """纯函数口径：CLI > 冒烟上限 > config；非冒烟时冒烟上限不生效。"""
        from prm.train_prm import _resolve_eval_max

        cfg = {"eval_max_samples": 500, "smoke_eval_max_samples": 64}
        assert _resolve_eval_max(cfg, smoke=False) == 500
        assert _resolve_eval_max(cfg, smoke=True) == 64        # 冒烟收紧
        assert _resolve_eval_max(cfg, smoke=True, cli_value=2) == 2
        assert _resolve_eval_max(cfg, smoke=False, cli_value=2) == 2
        assert _resolve_eval_max(cfg, smoke=True, cli_value=0) is None   # 0 = 全量
        assert _resolve_eval_max({"eval_max_samples": 10}, smoke=True) == 10  # 已更严则不放松
        assert _resolve_eval_max({}, smoke=False) is None

    def test_use_kernels_failure_falls_back(self, tiny_world, monkeypatch):
        """kernels 不可用时不得让训练起不来：告警 + 回退参考实现（fail-open）。"""
        import copy

        from prm import train_prm as T

        original = T.VerdictScorer.from_pretrained
        seen: list[bool] = []

        def fake(cls, *a, **kw):
            seen.append(bool(kw.get("use_kernels", False)))
            if kw.get("use_kernels"):
                raise ValueError("Kernels are not available. Please install ...")
            return original(*a, **kw)

        monkeypatch.setattr(T.VerdictScorer, "from_pretrained", classmethod(fake))
        cfg = copy.deepcopy(tiny_world["cfg"])
        cfg["train"]["use_kernels"] = True
        run_dir = train(cfg, "smoketest_kernfallback", smoke=True)
        manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        assert seen == [True, False]                 # 先试内核 → 失败 → 回退
        assert manifest["use_kernels"] is False
        assert manifest["best_metric"] is not None   # 回退后照常训完

    def test_use_kernels_kernelize_failure_after_move_falls_back(
            self, tiny_world, monkeypatch):
        """内核化在**搬设备后**才失败（HF Hub 不可达）也必须回退——真机实测路径。

        `from_pretrained(use_kernels=True)` 里的 kernelize 发生在加载设备上（CPU），
        真正换成 cuda 内核是 `to_device_and_kernelize` 那一步；实测本机 HF 不可达，
        异常正是在那里抛出（httpx.ConnectTimeout）。若它落在 try 之外，训练会直接挂。
        """
        import copy

        from prm import train_prm as T

        def fake_move(self, device):
            if getattr(self, "use_kernels", False):
                raise OSError("[Errno 110] Connection timed out")
            return self

        monkeypatch.setattr(T.VerdictScorer, "to_device_and_kernelize", fake_move)
        cfg = copy.deepcopy(tiny_world["cfg"])
        cfg["train"]["use_kernels"] = True
        run_dir = train(cfg, "smoketest_kernmove", smoke=True)
        manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        assert manifest["use_kernels"] is False
        assert manifest["best_metric"] is not None    # 回退后照常训完

    def test_reload_and_score(self, tiny_world):
        """训练产物可经 eval_prm.load_scorer_for_run 复原并前向打分。"""
        import torch
        from prm.eval_prm import load_scorer_for_run

        run_dir = tiny_world["tmp"] / "runs" / "smoketest"
        scorer, manifest, collator = load_scorer_for_run(str(run_dir), device="cpu")
        assert manifest["run"] == "smoketest"
        batch = collator([{"messages": make_tiny_messages(1), "label": 0.0}])
        p = scorer.predict_proba(batch["input_ids"], batch["attention_mask"])
        assert p.shape == (1,) and 0.0 <= float(p[0]) <= 1.0

    def test_class_weights_from_build_manifest(self, tiny_world):
        """存在 outputs/prm/manifest.json 时 w_neg 取 build 统计（一致性口径）。"""
        tmp = tiny_world["tmp"]
        (tmp / "manifest.json").write_text(
            json.dumps({"class_weights": {"w_pos": 1.0, "w_neg": 2.5}}), encoding="utf-8")
        run_dir = train(tiny_world["cfg"], "smoketest2", smoke=True)
        manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        assert manifest["w_neg"] == pytest.approx(2.5)

    def test_rejects_multi_sample_batch_size(self, tiny_world):
        """bs>1 会引入 padding → 必须直接拒绝（计划书 §14.2）。"""
        import copy

        cfg = copy.deepcopy(tiny_world["cfg"])
        cfg["train"]["batch"] = {"per_device_train_batch_size": 2,
                                 "gradient_accumulation_steps": 1}
        with pytest.raises(SystemExit, match="必须为 1"):
            train(cfg, "smoketest_bs2", smoke=True)

    def test_lora_gradient_checkpointing_roundtrip(self, tiny_world):
        """宿主 GPU 真实走的组合（LoRA + gradient_checkpointing）先在 CPU 打通。

        上面的冒烟用 ``lora=None / gradient_checkpointing=False``，与
        ``config/prm.yaml`` 的实际配置不同；GPU 路径在沙箱无法执行，
        这一组合的 API 问题只能在此暴露（与 GPU 门控测试被 skip 藏住的
        NameError 同类）。
        """
        import copy
        import json

        cfg = copy.deepcopy(tiny_world["cfg"])
        cfg["train"]["lora"] = {
            "r": 4, "alpha": 8, "dropout": 0.0, "bias": "none",
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                               "gate_proj", "up_proj", "down_proj"],
        }
        cfg["train"]["gradient_checkpointing"] = True
        run_dir = train(cfg, "smoketest_lora", smoke=True)

        # LoRA 产物：adapter 两件套 + 镜像的 model.safetensors（回载通用路径）
        assert (run_dir / "adapter_config.json").exists()
        assert (run_dir / "adapter_model.safetensors").exists()
        assert (run_dir / "model.safetensors").exists()
        manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        assert manifest["best_metric"] is not None      # dev 评估跑通
        assert manifest["lora"]["r"] == 4

        # adapter 确实被训练更新过（LoRA B 非全零），且 eval_prm 能回载打分
        from safetensors.torch import load_file
        state = load_file(str(run_dir / "model.safetensors"))
        assert any("lora_A" in k for k in state), "镜像的 model.safetensors 应含 LoRA 键"
        assert any(v.abs().sum().item() > 0 for k, v in state.items() if "lora_B" in k), \
            "LoRA B 全零说明训练没有真正更新 adapter"

        from prm.eval_prm import load_scorer_for_run
        scorer, _m, collator = load_scorer_for_run(str(run_dir), device="cpu")
        batch = collator([{"messages": make_tiny_messages(1), "label": 1.0}])
        p = scorer.predict_proba(batch["input_ids"], batch["attention_mask"])
        assert p.shape == (1,) and 0.0 <= float(p[0]) <= 1.0

    def test_resume_from_checkpoint(self, tiny_world):
        """--resume 从最新检查点续训（正式训练跑数小时，中断后不从头再来）。"""
        import copy

        run_dir = train(tiny_world["cfg"], "smoketest_resume", smoke=True)
        ckpts = sorted((run_dir / "checkpoints").glob("checkpoint-*"))
        assert ckpts, "冒烟（save_steps=1）应产生检查点"
        first = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        assert first["resumed_from"] is None

        cfg = copy.deepcopy(tiny_world["cfg"])
        run_dir2 = train(cfg, "smoketest_resume", smoke=True, resume=True)
        second = json.loads((run_dir2 / "run_manifest.json").read_text(encoding="utf-8"))
        assert second["resumed_from"] is not None
        assert "checkpoint-" in second["resumed_from"]


class TestReporting:
    """训练曲线记录（SwanLab）：默认关闭；开启后 env 翻译与落盘产物。"""

    @pytest.fixture(autouse=True)
    def _clean_swanlab(self, monkeypatch):
        """测试间清理 swanlab 全局 run 与 SWANLAB_* 环境变量，避免相互污染。"""
        for k in ("SWANLAB_MODE", "SWANLAB_LOG_DIR", "SWANLAB_PROJECT", "SWANLAB_API_KEY"):
            monkeypatch.delenv(k, raising=False)
        try:
            import swanlab
        except ImportError:
            yield
            return

        def _finish():
            try:
                swanlab.get_run()
            except Exception:
                return
            try:
                swanlab.finish()
            except Exception:
                pass

        _finish()
        yield
        _finish()

    def test_report_to_default_off(self, tmp_path):
        from prm.train_prm import _setup_reporting

        assert _setup_reporting({}, "r", tmp_path) == []
        assert _setup_reporting({"report_to": None}, "r", tmp_path) == []
        assert _setup_reporting({"report_to": []}, "r", tmp_path) == []
        assert _setup_reporting({"report_to": ["none"]}, "r", tmp_path) == ["none"]

    def test_swanlab_env_from_config(self, tmp_path):
        import os
        import prm.train_prm as tp

        # 上游坑：SWANLAB_PROJECT 只接受 JSON 对象，普通字符串会让 init 抛
        # SettingsError → 代码改为不写该环境变量，project 只经 init(project=...)
        os.environ["SWANLAB_PROJECT"] = "should-be-removed"
        try:
            got = tp._setup_reporting(
                {"report_to": ["swanlab"],
                 "swanlab": {"project": "CodeAgentRL", "workspace": "2410104018",
                             "mode": "disabled",
                             "log_dir": str(tmp_path / "sl")}},
                "run-x", tmp_path)
        finally:
            os.environ.pop("SWANLAB_PROJECT", None)
        assert got == ["swanlab"]
        assert os.environ["SWANLAB_MODE"] == "disabled"
        assert os.environ["SWANLAB_LOG_DIR"] == str(tmp_path / "sl")
        assert "SWANLAB_PROJECT" not in os.environ, "不应保留普通字符串形式的 project"
        assert tp._SWANLAB_PROJECT == "CodeAgentRL"
        assert tp._SWANLAB_WORKSPACE == "2410104018"

    def test_swanlab_cloud_without_login_disables(self, tmp_path, monkeypatch):
        """cloud 模式但无 key、无任何登录凭据 → 禁用而非交互登录阻塞训练。"""
        import prm.train_prm as tp

        monkeypatch.delenv("SWANLAB_API_KEY", raising=False)
        monkeypatch.setattr(tp, "_swanlab_credential_paths",
                            lambda: [tmp_path / "no-such.netrc"])
        got = tp._setup_reporting(
            {"report_to": ["swanlab"],
             "swanlab": {"project": "CodeAgentRL", "mode": "cloud"}}, "run-y", tmp_path)
        assert got == [], "未登录时应摘掉 swanlab，避免 init 卡在交互登录"

    def test_swanlab_cloud_init_args(self, tmp_path, monkeypatch):
        """已登录（凭据存在）→ 按官方流程 init：project/workspace/name/config。"""
        swanlab = pytest.importorskip("swanlab")
        import prm.train_prm as tp

        cred = tmp_path / ".swanlab" / ".netrc"
        cred.parent.mkdir(parents=True, exist_ok=True)
        cred.write_text("machine https://api.swanlab.cn login x password y\n",
                        encoding="utf-8")
        monkeypatch.delenv("SWANLAB_API_KEY", raising=False)
        monkeypatch.setattr(tp, "_swanlab_credential_paths", lambda: [cred])
        calls: dict = {}
        monkeypatch.setattr(swanlab, "init", lambda **kw: calls.update(kw))

        got = tp._setup_reporting(
            {"report_to": ["swanlab"],
             "swanlab": {"project": "CodeAgentRL", "workspace": "2410104018",
                         "mode": "cloud"}},
            "m4-v1", tmp_path)
        assert got == ["swanlab"]
        assert calls["project"] == "CodeAgentRL"
        assert calls["workspace"] == "2410104018"
        assert calls["name"] == "m4-v1"
        assert calls["mode"] == "cloud"
        assert calls["config"]["run"] == "m4-v1"

    def test_swanlab_missing_package_fails_loudly(self, tmp_path, monkeypatch):
        import types
        import prm.train_prm as tp

        monkeypatch.setattr(tp, "importlib",
                            types.SimpleNamespace(util=types.SimpleNamespace(
                                find_spec=lambda name: None)))
        with pytest.raises(SystemExit, match="pip install swanlab"):
            tp._setup_reporting({"report_to": ["swanlab"]}, "r", tmp_path)

    def test_swanlab_local_end_to_end(self, tiny_world):
        """local 模式（无网络/无 key）真跑一次训练循环：env → 回调 → 落盘 → manifest。"""
        pytest.importorskip("swanlab")
        import copy

        cfg = copy.deepcopy(tiny_world["cfg"])
        log_dir = tiny_world["tmp"] / "swanlog"
        cfg["train"]["report_to"] = ["swanlab"]
        cfg["train"]["swanlab"] = {"project": "prm-test", "mode": "local",
                                   "log_dir": str(log_dir)}
        run_dir = train(cfg, "smoketest_swanlab", smoke=True)

        man = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        assert man["report_to"] == ["swanlab"]
        assert man["swanlab"]["project"] == "prm-test"
        assert man["swanlab"]["mode"] == "local"
        files = [p for p in log_dir.rglob("*") if p.is_file()]
        assert files, f"local 模式应在 {log_dir} 落盘 swanlog 文件"
