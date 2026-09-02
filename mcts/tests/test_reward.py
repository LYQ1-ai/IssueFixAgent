# SPDX-License-Identifier: BSD-3-Clause

"""``mcts.reward`` diff 解析 + 三粒度定位 F1 测试（纯逻辑）。"""

import pytest

from mcts.instances import Gold
from mcts.reward import (
    compute_file_f1_score,
    is_diff_like,
    parse_diff_locations,
    patch_localization_f1,
    reward_from_patch,
)

PATCH_WITH_CLASS = """diff --git a/src/pptx/chart/plot.py b/src/pptx/chart/plot.py
index 1111111..2222222 100644
--- a/src/pptx/chart/plot.py
+++ b/src/pptx/chart/plot.py
@@ -40,6 +40,8 @@ class PlotTypeInspector:
         self._plot_types = {}
         return self._plot_types

+    def _differentiate_xy_chart_type(self, chart):
+        return "XY"
+
@@ -90,7 +92,8 @@ def make_plot(chart):
         raise ValueError(chart.plot_type)
+
+    return plot
"""

PATCH_NEW_FILE = """diff --git a/README.md b/README.md
new file mode 100644
index 0000000..1111111
--- /dev/null
+++ b/README.md
@@ -0,0 +1,3 @@
+# title
+text
"""


class TestParseDiffLocations:
    def test_files_from_headers(self):
        files, _, _ = parse_diff_locations(PATCH_WITH_CLASS)
        assert files == {"src/pptx/chart/plot.py"}

    def test_class_and_method_entities(self):
        files, modules, entities = parse_diff_locations(PATCH_WITH_CLASS)
        # class hunk：类内方法归属类；def hunk（新 hunk 重置类上下文）：顶层函数
        assert "src/pptx/chart/plot.py:PlotTypeInspector" in modules
        assert "src/pptx/chart/plot.py:PlotTypeInspector._differentiate_xy_chart_type" in entities
        assert "src/pptx/chart/plot.py:make_plot" in entities

    def test_new_file_ignored_if_not_python(self):
        files, modules, entities = parse_diff_locations(PATCH_NEW_FILE)
        assert files == {"README.md"}
        assert modules == set()
        assert entities == set()

    def test_empty_patch(self):
        assert parse_diff_locations("") == (set(), set(), set())


class TestF1:
    def test_compute_file_f1(self):
        assert compute_file_f1_score({"a", "b"}, {"a", "c"}) == 0.5
        assert compute_file_f1_score(set(), {"a"}) == 0.0
        assert compute_file_f1_score({"a"}, set()) == 0.0   # gt 空 → 0
        assert compute_file_f1_score({"a", "b"}, {"a", "b"}) == 1.0

    def test_patch_localization_f1_sum(self):
        gold = Gold(files=frozenset({"src/pptx/chart/plot.py"}),
                    modules=frozenset({"src/pptx/chart/plot.py:PlotTypeInspector",
                                       "src/pptx/chart/plot.py:make_plot"}),
                    entities=frozenset({"src/pptx/chart/plot.py:PlotTypeInspector._differentiate_xy_chart_type",
                                        "src/pptx/chart/plot.py:make_plot"}))
        reward, details = patch_localization_f1(PATCH_WITH_CLASS, gold)
        assert reward == 3.0  # 三层全命中
        assert details["file_f1"] == 1.0

    def test_partial_match(self):
        gold = Gold(files=frozenset({"src/pptx/chart/plot.py"}),
                    modules=frozenset({"src/pptx/chart/plot.py:OtherClass"}),
                    entities=frozenset({"src/pptx/chart/plot.py:unrelated"}))
        reward, details = patch_localization_f1(PATCH_WITH_CLASS, gold)
        assert reward == 1.0  # 只有 file 层命中

    def test_class_attribution_penalty_for_methods(self):
        # gold 只含类方法（diff 解析不出类归属 → 归为顶层函数，recall 满分但 precision 减半）
        gold = Gold(files=frozenset({"src/pptx/chart/plot.py"}),
                    modules=frozenset({"src/pptx/chart/plot.py:PlotTypeInspector"}),
                    entities=frozenset({"src/pptx/chart/plot.py:PlotTypeInspector._differentiate_xy_chart_type"}))
        reward, details = patch_localization_f1(PATCH_WITH_CLASS, gold)
        # file 1.0 + module F1(1/2, 1)=2/3 + entity F1(1/2, 1)=2/3
        assert reward == pytest.approx(1.0 + 2 / 3 + 2 / 3)

    def test_empty_patch_zero(self):
        reward, details = patch_localization_f1("", Gold())
        assert reward == 0.0
        assert details["reason"] == "empty_patch"


class TestRewardFromPatch:
    def test_threshold(self):
        gold = Gold(files=frozenset({"src/pptx/chart/plot.py"}),
                    modules=frozenset({"src/pptx/chart/plot.py:PlotTypeInspector",
                                       "src/pptx/chart/plot.py:make_plot"}),
                    entities=frozenset({"src/pptx/chart/plot.py:PlotTypeInspector._differentiate_xy_chart_type",
                                        "src/pptx/chart/plot.py:make_plot"}))
        reward, correct, details = reward_from_patch(PATCH_WITH_CLASS, gold, threshold=0.5)
        assert reward == 3.0 and correct is True

    def test_zero_patch_false(self):
        reward, correct, _ = reward_from_patch("", Gold())
        assert reward == 0.0 and correct is False


class TestIsDiffLike:
    def test_detection(self):
        assert is_diff_like("diff --git a/x.py b/x.py\n+++ b/x.py\n")
        assert not is_diff_like("just a summary of changes")
