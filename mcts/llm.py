# SPDX-License-Identifier: BSD-3-Clause

"""模型路由层（PLAN §2.1）。

- :func:`build_model_config`：构建 mini-swe-agent ``get_model`` 用的 model_config
  （``cost_tracking: ignore_errors`` 关闭本地模型的成本核算报错；``model_kwargs``
  里带 ``api_base`` / ``api_key`` / ``temperature``，对齐 ``scripts/demo.py`` 的
  已验证配置）。Agent 执行路径（RepoAgent / ReplayRunner）统一从这里取配置。
- :class:`CompletionClient`：**备用**纯文本补全客户端（litellm 直连，指数退避重试
  429 / timeout），供后续"强生成器重标注"（ReARTeR my_config_2.yaml 思路）或
  其它非 Agent 补全使用；当前 MCTS 引擎不依赖它。

版本约束：``openai==2.54.0`` / ``litellm==1.97.0``（README §7 锁定），不升级。
litellm 惰性导入：不 import 本模块的纯逻辑测试不会拉起 litellm。
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any, Optional

logger = logging.getLogger("mcts.llm")

# 本机部署的 vLLM 默认地址（A800：GPU0 起 gemma-4，见 PLAN §0）
DEFAULT_BASE_URL = "http://localhost:8010/v1"
# ReARTeR 的 rollout 多样性来源：temperature 从 [0.7, 1.0] 随机采样（见 docs/01 §10.1）
DEFAULT_TEMPERATURE_RANGE = (0.7, 1.0)


def build_model_config(
    model_name: str,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: Optional[float] = None,
    *,
    cost_tracking: str = "ignore_errors",
    model_class: Optional[str] = None,
) -> dict:
    """构建 mini-swe-agent 的 model_config（与 scripts/demo.py 已验证路径一致）。

    Args:
        model_name: 需带 provider 前缀（如 ``openai/gemma-4``；裸名会报
            "LLM Provider NOT provided"，README §4）。
        base_url: OpenAI 兼容服务地址（本地 vLLM）；None 表示 litellm 默认路由。
        api_key: API key；本地服务不校验，给占位符即可。
        temperature: 本 rollout 的采样温度（None = 让服务端用默认值）。
        cost_tracking: 本地模型无成本表时用 ``ignore_errors`` 关闭报错。
        model_class: 自定义 model 类的 import 路径（如
            ``agent.submit_model.SubmitLocationsModel`` —— 工具列表变为
            ``[bash, submit_locations]``）；None = 官方 ``LitellmModel``
            （``tools=[bash]``）。由 ``minisweagent.models.get_model`` 解析。
    """
    model_kwargs: dict[str, Any] = {}
    if base_url:
        model_kwargs["api_base"] = base_url
        model_kwargs["api_key"] = api_key if api_key is not None else "local-rollout"
    elif api_key:
        model_kwargs["api_key"] = api_key
    if temperature is not None:
        model_kwargs["temperature"] = temperature
    cfg: dict[str, Any] = {"cost_tracking": cost_tracking, "model_kwargs": model_kwargs}
    if model_class:
        cfg["model_class"] = model_class
    return cfg


class CompletionClient:
    """备用纯文本补全客户端（PLAN §2.1；当前引擎不依赖）。

    与 Agent 路径无关：直接调用 ``litellm.completion`` 完成一次文本补全，
    带指数退避重试（429 / timeout / 5xx）。记录调用次数供预算统计。
    """

    def __init__(
        self,
        model_name: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        *,
        retries: int = 3,
        backoff_base: float = 2.0,
    ):
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.retries = retries
        self.backoff_base = backoff_base
        self.n_calls = 0

    def complete(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """单次补全；失败按指数退避重试，重试耗尽抛异常。"""
        import litellm  # 惰性导入（纯逻辑测试不拉起）

        kwargs: dict[str, Any] = {"model": self.model_name, "messages": messages,
                                  "temperature": temperature}
        if self.base_url:
            kwargs["api_base"] = self.base_url
            kwargs["api_key"] = self.api_key or "local-rollout"
        elif self.api_key:
            kwargs["api_key"] = self.api_key
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        last_exc: Optional[Exception] = None
        for attempt in range(self.retries):
            try:
                resp = litellm.completion(**kwargs)
                self.n_calls += 1
                return str(resp.choices[0].message.content or "")
            except Exception as e:  # noqa: BLE001 - 统一按可重试错误处理
                last_exc = e
                status = getattr(e, "status_code", None)
                if status not in (None, 429, 500, 502, 503, 504):
                    raise
                delay = self.backoff_base ** attempt + random.uniform(0, 0.5)
                logger.warning("completion attempt %d failed (%s); retry in %.1fs",
                               attempt + 1, type(e).__name__, delay)
                time.sleep(delay)
        raise RuntimeError(f"completion failed after {self.retries} retries: {last_exc}")


__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_TEMPERATURE_RANGE",
    "build_model_config",
    "CompletionClient",
]
