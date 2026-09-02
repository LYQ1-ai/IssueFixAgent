"""``mcts/instances.py``（M0 数据预处理：读取 / 过滤 / GT 抽取）测试。

对齐 codescout 数据处理部分（build_dataset.py 直读 + file_localization.py
的 multilevel_localization_f1_reward GT 解析口径）。纯单元测试，无 Docker / 网络；
真实数据数量级对照用 ``test_real_swe_smith_*``（数据文件存在时运行，否则跳过）。
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# 保证 ``mcts`` 包可导入（无论从项目根还是 test/ 目录启动）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcts.instances import (  # noqa: E402
    Gold,
    Instance,
    build_dataset,
    extract_gold,
    filter_instances,
    instances_from_parquet,
    is_python_path,
    parse_repo,
    patch_creates_or_deletes_files,
    read_instances,
)

DATA_DIR = (
    Path(__file__).resolve().parents[1]
    / "ref_papers" / "codescout" / "data" / "swe_smith"
)
TRAIN_PARQUET = DATA_DIR / "train.parquet"

_DEFAULT_PATCH = (
    "diff --git a/src/cli.py b/src/cli.py\n"
    "index 1111111..2222222 100644\n"
    "--- a/src/cli.py\n"
    "+++ b/src/cli.py\n"
    "@@ -1,3 +1,3 @@\n"
    "-old\n"
    "+new\n"
)


def make_instance(**overrides) -> Instance:
    base = dict(
        instance_id="owner__repo.deadbeef.func_basic__task1",
        repo="swesmith/owner__repo.deadbeef",
        owner="swesmith",
        name="owner__repo",
        commit8="deadbeef",
        base_commit=None,
        problem_statement="Fix the bug in the CLI option parsing.",
        patch=_DEFAULT_PATCH,
        use_patch=True,
        gold=Gold(
            frozenset(["src/cli.py"]),
            frozenset(["src/cli.py:cli"]),
            frozenset(["src/cli.py:cli"]),
        ),
        source="train",
    )
    base.update(overrides)
    return Instance(**base)


def make_file_changes(*changes: dict) -> list[dict]:
    return list(changes)


# ---------------------------------------------------------------------------
# parse_repo / is_python_path
# ---------------------------------------------------------------------------


class TestParseRepo:
    def test_swesmith_repo(self):
        assert parse_repo("swesmith/theskumar__python-dotenv.2b8635b7") == (
            "swesmith", "theskumar__python-dotenv", "2b8635b7",
        )

    def test_plain_repo(self):
        assert parse_repo("django/django") == ("django", "django", None)

    def test_git_suffix_stripped(self):
        assert parse_repo("django/django.git") == ("django", "django", None)

    def test_missing_owner_raises(self):
        with pytest.raises(ValueError):
            parse_repo("just-a-name")

    def test_repo_without_commit8(self):
        assert parse_repo("swesmith/foo__bar") == ("swesmith", "foo__bar", None)


class TestIsPythonPath:
    def test_py_file(self):
        assert is_python_path("src/dotenv/cli.py")

    def test_module_entity_form(self):
        assert is_python_path("src/dotenv/cli.py:cli")
        assert is_python_path("src/pptx/chart/plot.py:PlotTypeInspector")

    def test_non_python(self):
        assert not is_python_path("README.md")
        assert not is_python_path("docs/guide.rst")
        assert not is_python_path("setup.cfg:section")


# ---------------------------------------------------------------------------
# extract_gold（对齐 codescout multilevel_localization_f1_reward 的 GT 口径）
# ---------------------------------------------------------------------------


class TestExtractGold:
    def test_basic_three_levels(self):
        fc = make_file_changes({
            "file": "src/dotenv/cli.py",
            "changes": {
                "edited_modules": ["src/dotenv/cli.py:cli"],
                "edited_entities": ["src/dotenv/cli.py:cli"],
                "added_modules": None,
                "added_entities": None,
            },
        })
        gold = extract_gold(fc)
        assert gold.files == frozenset({"src/dotenv/cli.py"})
        assert gold.modules == frozenset({"src/dotenv/cli.py:cli"})
        assert gold.entities == frozenset({"src/dotenv/cli.py:cli"})

    def test_none_added_no_crash(self):
        fc = make_file_changes({
            "file": "a.py",
            "changes": {
                "edited_modules": None,
                "edited_entities": None,
                "added_modules": None,
                "added_entities": None,
            },
        })
        gold = extract_gold(fc)
        assert gold.files == frozenset({"a.py"})
        assert gold.modules == frozenset() and gold.entities == frozenset()

    def test_added_included_by_default(self):
        fc = make_file_changes({
            "file": "a.py",
            "changes": {
                "edited_modules": ["a.py:A"],
                "added_modules": ["a.py:B"],
                "edited_entities": ["a.py:f"],
                "added_entities": ["a.py:g"],
            },
        })
        gold = extract_gold(fc)
        assert gold.modules == frozenset({"a.py:A", "a.py:B"})
        assert gold.entities == frozenset({"a.py:f", "a.py:g"})

    def test_added_excluded_when_disabled(self):
        fc = make_file_changes({
            "file": "a.py",
            "changes": {
                "edited_modules": ["a.py:A"],
                "added_modules": ["a.py:B"],
                "edited_entities": ["a.py:f"],
                "added_entities": ["a.py:g"],
            },
        })
        gold = extract_gold(fc, include_added_modules=False, include_added_entities=False)
        assert gold.modules == frozenset({"a.py:A"})
        assert gold.entities == frozenset({"a.py:f"})

    def test_ndarray_input_like_real_parquet(self):
        # 真实 parquet 里 file_changes 是 ndarray(dtype=object)，元素是 dict
        fc = np.array(
            [{"file": "a.py", "changes": {"edited_entities": np.array(["a.py:f"], dtype=object)}}],
            dtype=object,
        )
        gold = extract_gold(fc)
        assert gold.files == frozenset({"a.py"})
        assert gold.entities == frozenset({"a.py:f"})

    def test_python_only_filters_non_python(self):
        fc = make_file_changes(
            {"file": "a.py", "changes": {"edited_modules": ["a.py:A"]}},
            {"file": "README.md", "changes": {"edited_modules": ["README.md:title"]}},
        )
        gold = extract_gold(fc, python_only=True)
        assert gold.files == frozenset({"a.py"})
        assert "README.md:title" not in gold.modules

    def test_empty(self):
        assert extract_gold([]).is_empty()
        assert extract_gold(None).is_empty()
        assert extract_gold(make_file_changes({"file": "README.md", "changes": {}})).is_empty()


# ---------------------------------------------------------------------------
# patch_creates_or_deletes_files（过滤规则 2）
# ---------------------------------------------------------------------------


class TestPatchCreatesOrDeletesFiles:
    def test_new_file_mode(self):
        assert patch_creates_or_deletes_files("diff --git a/x.py b/x.py\nnew file mode 100644\n")

    def test_deleted_file_mode(self):
        assert patch_creates_or_deletes_files("diff --git a/x.py b/x.py\ndeleted file mode 100644\n")

    def test_dev_null(self):
        assert patch_creates_or_deletes_files("--- /dev/null\n+++ b/x.py\n")
        assert patch_creates_or_deletes_files("--- a/x.py\n+++ /dev/null\n")

    def test_plain_diff(self):
        assert not patch_creates_or_deletes_files(_DEFAULT_PATCH)


# ---------------------------------------------------------------------------
# filter_instances（PLAN §1.2 五条规则）
# ---------------------------------------------------------------------------


class TestFilterInstances:
    def test_valid_instance_kept(self):
        kept, report = filter_instances([make_instance()])
        assert len(kept) == 1
        assert report["original_total"] == 1
        assert report["kept_total"] == 1
        assert report["rules"] == {}

    def test_empty_statement_dropped(self):
        inst = make_instance(problem_statement="   \n\t ")
        kept, report = filter_instances([inst])
        assert kept == []
        assert report["rules"]["empty_statement"] == 1

    def test_statement_length_bounds(self):
        cfg = {"filter": {"min_statement_len": 50, "max_statement_len": 100}}
        short = make_instance(problem_statement="short")
        long = make_instance(problem_statement="x" * 200)
        ok = make_instance(problem_statement="y" * 75)
        kept, report = filter_instances([short, long, ok], cfg)
        assert [i.instance_id for i in kept] == [ok.instance_id]
        assert report["rules"] == {"statement_too_short": 1, "statement_too_long": 1}

    def test_created_deleted_files_dropped(self):
        inst = make_instance(patch="diff --git a/new.py b/new.py\nnew file mode 100644\n")
        kept, report = filter_instances([inst])
        assert kept == []
        assert report["rules"]["created_or_deleted_files"] == 1

    def test_no_valid_gold_dropped(self):
        inst = make_instance(gold=Gold())
        kept, report = filter_instances([inst])
        assert kept == []
        assert report["rules"]["no_valid_gold"] == 1

    def test_duplicate_instance_id_keeps_first(self):
        a = make_instance(instance_id="dup", problem_statement="first")
        b = make_instance(instance_id="dup", problem_statement="second")
        kept, report = filter_instances([a, b])
        assert [i.problem_statement for i in kept] == ["first"]
        assert report["rules"]["duplicate_instance_id"] == 1

    def test_invalid_format_dropped(self):
        inst = make_instance(repo="no-slash-here", owner="")
        kept, report = filter_instances([inst])
        assert kept == []
        assert report["rules"]["invalid_format"] == 1

    def test_report_repo_counts(self):
        insts = [
            make_instance(instance_id=f"r{i}__t", repo="swesmith/r%d.deadbeef" % i)
            for i in range(3)
        ]
        insts.append(make_instance(instance_id="r0__t", repo="swesmith/r0.deadbeef"))  # 真重复
        kept, report = filter_instances(insts)
        assert report["n_repos_original"] == 3
        assert report["n_repos_kept"] == 3  # 重复实例的 repo 仍在
        assert report["rules"] == {"duplicate_instance_id": 1}
        assert report["dropped_total"] == 1

    def test_disabled_rules_keep_everything(self):
        cfg = {"filter": {k: False for k in (
            "drop_empty_statement", "drop_created_deleted_files", "drop_no_valid_gold",
            "drop_duplicates", "drop_invalid_format", "min_statement_len", "max_statement_len",
        )}}
        insts = [
            make_instance(problem_statement="  "),
            make_instance(patch="new file mode 100644\n"),
            make_instance(gold=Gold()),
            make_instance(instance_id="dup"),
            make_instance(instance_id="dup"),
            make_instance(repo="bad", owner=""),
        ]
        kept, _ = filter_instances(insts, cfg)
        assert len(kept) == 6


# ---------------------------------------------------------------------------
# read_instances / build_dataset（synthetic parquet + 往返）
# ---------------------------------------------------------------------------


def _write_synthetic_parquet(path: Path, rows: list[dict]) -> None:
    """写一个 codescout 风格的 parquet；file_changes 以 JSON 字符串落盘
    （read_instances 的 _as_list 支持 str 形态，pyarrow 无序列化问题）。"""
    df = pd.DataFrame(
        {
            "instance_id": [r["instance_id"] for r in rows],
            "file_changes": [json.dumps(r["file_changes"]) for r in rows],
            "repo": [r["repo"] for r in rows],
            "base_commit": [r.get("base_commit") for r in rows],
            "problem_statement": [r["problem_statement"] for r in rows],
            "patch": [r["patch"] for r in rows],
            "use_patch": [r.get("use_patch", True) for r in rows],
            "prompt": [r.get("prompt", []) for r in rows],
        }
    )
    df.to_parquet(path, index=False)


def _synthetic_rows():
    return [
        {
            "instance_id": "theskumar__python-dotenv.2b8635b7.func_basic__n6cxbsay",
            "repo": "swesmith/theskumar__python-dotenv.2b8635b7",
            "base_commit": None,
            "problem_statement": "CLI options getting mixed up.",
            "patch": _DEFAULT_PATCH,
            "file_changes": [
                {
                    "file": "src/dotenv/cli.py",
                    "changes": {
                        "edited_modules": ["src/dotenv/cli.py:cli"],
                        "edited_entities": ["src/dotenv/cli.py:cli"],
                        "added_modules": None,
                        "added_entities": None,
                    },
                }
            ],
        },
        {
            "instance_id": "kurtmckee__feedparser.cad965a3.func_basic__abc",
            "repo": "swesmith/kurtmckee__feedparser.cad965a3",
            "base_commit": None,
            "problem_statement": "Parser crashes on empty feed.",
            "patch": _DEFAULT_PATCH,
            "file_changes": [
                {
                    "file": "feedparser/parser.py",
                    "changes": {
                        "edited_modules": ["feedparser/parser.py:FeedParser"],
                        "edited_entities": ["feedparser/parser.py:FeedParser._parse"],
                        "added_modules": None,
                        "added_entities": None,
                    },
                }
            ],
        },
    ]


class TestReadInstances:
    def test_read_parquet(self, tmp_path):
        path = tmp_path / "train.parquet"
        _write_synthetic_parquet(path, _synthetic_rows())
        insts = read_instances(path)
        assert len(insts) == 2
        first = insts[0]
        assert first.instance_id == "theskumar__python-dotenv.2b8635b7.func_basic__n6cxbsay"
        assert (first.owner, first.name, first.commit8) == (
            "swesmith", "theskumar__python-dotenv", "2b8635b7",
        )
        assert first.base_commit is None
        assert first.use_patch is True
        assert first.source == "train"  # 缺省取文件名
        assert first.gold.files == frozenset({"src/dotenv/cli.py"})
        assert first.gold.entities == frozenset({"src/dotenv/cli.py:cli"})
        assert first.repo_url == "https://github.com/swesmith/theskumar__python-dotenv.2b8635b7.git"

    def test_read_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_instances(tmp_path / "nope.parquet")

    def test_read_missing_column_raises(self, tmp_path):
        path = tmp_path / "bad.parquet"
        pd.DataFrame({"instance_id": ["x"]}).to_parquet(path)
        with pytest.raises(ValueError, match="missing required columns"):
            read_instances(path)

    def test_read_invalid_repo_row(self, tmp_path):
        path = tmp_path / "train.parquet"
        rows = _synthetic_rows()
        rows[0]["repo"] = "no-slash"  # 非法格式 -> owner 为空，过滤阶段剔除
        _write_synthetic_parquet(path, rows)
        insts = read_instances(path)
        assert insts[0].owner == "" and insts[0].name == "no-slash"


class TestBuildDataset:
    def test_build_and_roundtrip(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        _write_synthetic_parquet(data_dir / "train.parquet", _synthetic_rows())
        out = tmp_path / "out"
        report = build_dataset(
            [data_dir / "train.parquet"], output_dir=out, cfg={}
        )
        assert report["original_total"] == 2
        assert report["kept_total"] == 2
        assert report["n_repos_kept"] == 2
        assert report["source_counts"] == {"train": 2}
        inst_path = out / "instances.parquet"
        assert inst_path.is_file()
        assert (out / "data_report.json").is_file()
        assert (out / "data_report.md").is_file()

        # 往返：从 instances.parquet 读回，gold 与字段一致
        insts = instances_from_parquet(inst_path)
        assert len(insts) == 2
        assert insts[0].gold.files == frozenset({"src/dotenv/cli.py"})
        assert insts[0].gold.entities == frozenset({"src/dotenv/cli.py:cli"})
        assert insts[0].repo == "swesmith/theskumar__python-dotenv.2b8635b7"
        assert insts[0].source == "train"

    def test_build_report_markdown_contains_counts(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        _write_synthetic_parquet(data_dir / "train.parquet", _synthetic_rows())
        report = build_dataset(
            [data_dir / "train.parquet"],
            output_dir=tmp_path / "out",
            cfg={"filter": {"max_statement_len": 10}},
        )
        assert report["rules"]["statement_too_long"] == 2
        md = (tmp_path / "out" / "data_report.md").read_text(encoding="utf-8")
        assert "statement_too_long" in md
        assert "0 -> " not in md  # 有剔除数的表格行存在


# ---------------------------------------------------------------------------
# 真实数据数量级对照（train.parquet 存在时运行）
# ---------------------------------------------------------------------------

_real_data = pytest.mark.skipif(
    not TRAIN_PARQUET.is_file(),
    reason="swe_smith/train.parquet not present locally",
)


class TestRealSweSmith:
    @_real_data
    def test_read_counts_match_codescout(self):
        insts = read_instances(TRAIN_PARQUET)
        assert len(insts) == 39187  # 对齐 codescout build_dataset 的训练集规模
        assert len({i.repo for i in insts}) == 131
        assert all(i.base_commit is None for i in insts)
        assert all(i.use_patch for i in insts)

    @_real_data
    def test_default_filter_drops_only_overlong(self):
        insts = read_instances(TRAIN_PARQUET)
        kept, report = filter_instances(insts)
        assert report["rules"] == {"statement_too_long": 3}  # 实测 3 行 >20000
        assert len(kept) == 39184
        assert report["n_repos_kept"] == 131

    @_real_data
    def test_no_length_limit_keeps_all(self):
        insts = read_instances(TRAIN_PARQUET)
        kept, report = filter_instances(
            insts, {"filter": {"max_statement_len": None}}
        )
        assert len(kept) == 39187 and report["dropped_total"] == 0

    @_real_data
    def test_gold_three_levels_and_added_entities(self):
        insts = read_instances(TRAIN_PARQUET)
        with_entities = sum(1 for i in insts if i.gold.entities)
        with_modules = sum(1 for i in insts if i.gold.modules)
        assert with_entities > 0 and with_modules > 0
        # 所有 gold 文件条目都应是 Python 文件
        for i in insts:
            assert all(f.endswith(".py") for f in i.gold.files)
        # 找到一条 added_entities 非空的实例，验证其并入 gold.entities
        df = pd.read_parquet(TRAIN_PARQUET)
        target = None
        for record in df.to_dict("records"):
            for change in record["file_changes"].tolist():
                ch = change.get("changes") or {}
                if ch.get("added_entities") is not None:
                    target = record["instance_id"]
                    break
            if target:
                break
        assert target is not None
        inst = next(i for i in insts if i.instance_id == target)
        record = next(r for r in df.to_dict("records") if r["instance_id"] == target)
        added_all = set()
        for change in record["file_changes"].tolist():
            ch = change.get("changes") or {}
            added_all.update(ch.get("added_entities") or [])
        assert added_all and added_all <= set(inst.gold.entities)
