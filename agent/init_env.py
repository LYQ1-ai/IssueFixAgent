# SPDX-License-Identifier: BSD-3-Clause

"""Agent 执行环境（Docker 容器）工厂：**创建即用、用完即销毁，不复用、无缓存**。

（v2 语义，2026-08-27 起）MCTS 高并发 Rollout 场景下每个 rollout 1–5 分钟、容器
创建 3–8 秒，复用带来的状态复位与污染风险不划算；容器生命周期改为：

    container = mgr.get_env(repo, commit)     # 每次调用都新建一个独立容器
    # ... 外部执行 rollout ...
    mgr.release_env(container)                # docker rm -f 删除（幂等）

- **每次 ``get_env`` 都创建新容器**（容器名含 uuid，天然互不冲突），不做
  ``(repo, commit) -> 容器`` 缓存，也不做存活校验/重启/复用；
- **仓库材料化仍然走本地 zip 缓存**（宿主机 clone 打成 zip → docker cp + 解压，
  跳过容器内 git clone）：这是仓库缓存，不是容器复用，保持不变；
- **并发创建限流**由调用方（``mcts.tasks.EnvFactory``）控制：只限同时创建的容器数，
  不限总创建次数（防 50 路 worker 同时 docker run + unzip 的瞬时 IO 风暴）；
- 创建失败自动清理半成品容器并抛错；``release_env`` 对未知/空容器名安全（幂等）。

环境准备流程（不变）：基于基础镜像启动常驻容器（``sleep infinity``）→ 容器内
确保 git / unzip / ripgrep（缺失走 apt，默认强制 IPv4）→ 仓库放入容器
（zip 缓存路径：docker cp + 解压；回退路径：容器内 git clone）→ ``git checkout``
切换 commit（**commit=None 时不切换**，适配 SWE-Smith 快照仓库）→ 可选
``git apply`` 应用 patch（bug 引入 diff）。

用法::

    from agent.init_env import EnvManager

    mgr = EnvManager()
    container = mgr.get_env("django/django", "6da8c1f0c46a8a0f1b8a")  # 每次新建
    # ... 外部 Agent 在 container 里执行任务 ...
    mgr.release_env(container)  # docker rm -f（幂等）

    或者使用模块级默认管理器：

    from agent.init_env import get_env, release_env
    container = get_env("django/django", "6da8c1f0c46a8a0f1b8a")
    release_env(container)

命令行用法::

    python -m agent.init_env get owner/repo --commit <sha>   # 打印新容器名
    python -m agent.init_env release <container-name>
    python -m agent.init_env cache owner/repo                # 生成仓库 zip 缓存
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
import uuid
import zipfile
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger("agent.init_env")

_INVALID_NAME_CHARS = re.compile(r"[^a-zA-Z0-9_.-]")


def _sanitize(name: str) -> str:
    """把仓库名 / commit 转成 Docker 容器名可用的 token（小写字母数字 . _ -）。"""
    name = name.replace("/", "__")
    name = _INVALID_NAME_CHARS.sub("_", name)
    return name.strip("._-")[:60] or "repo"


def _normalize_repo(repo_name: str) -> str:
    repo_name = repo_name.strip()
    if repo_name.endswith(".git"):
        repo_name = repo_name[:-4]
    if "/" not in repo_name:
        raise ValueError(f"repo_name must be in 'owner/repo' format, got: {repo_name!r}")
    return repo_name


def _env_bool(name: str, default: bool) -> bool:
    """读取布尔型环境变量（1/true/yes/on -> True，其余 -> False）。"""
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    """读取整型环境变量；未设置或非法时返回默认值。"""
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value.strip())
    except ValueError:
        logger.warning("Invalid integer for env %s=%r, using default %d", name, value, default)
        return default


@dataclass
class EnvConfig:
    """环境（容器）创建配置。

    所有字段都有代码内默认值，且均可用环境变量覆盖（见项目根目录
    ``.env_template``）。显式传入构造参数的优先级高于环境变量。

    Args:
        image: 基础镜像（env ``CODEAGENTRL_IMAGE``），默认 ``python:3.11-slim``
            （与 codescout 的最小终端环境对齐）。
        workdir: 容器内仓库挂载/克隆路径（env ``CODEAGENTRL_WORKDIR``），默认 ``/repo``。
        user: 可选的容器内用户（``uid:gid`` 或用户名，env ``CODEAGENTRL_USER``），
            默认 root。
        docker_executable: docker CLI 路径（env ``MSWEA_DOCKER_EXECUTABLE``）。
        install_ripgrep: 是否在容器内安装 ripgrep（env ``CODEAGENTRL_INSTALL_RIPGREP``，
            codescout agent 的核心搜索工具）。
        apt_force_ipv4: 是否强制容器内 apt 走 IPv4（env ``CODEAGENTRL_APT_FORCE_IPV4``，
            默认 true；规避 IPv6 路由不可达导致 apt 挂起的问题）。
        repo_cache_dir: 本地仓库 zip 缓存目录（env ``CODEAGENTRL_REPO_CACHE_DIR``）。
            配置后：创建容器前若 ``<repo 名（/ 替换为 __）>.zip`` 不存在，会先
            在宿主机 git clone 并打成 zip 存入缓存，再统一走 docker cp + 解压 +
            checkout（与缓存命中路径一致）；未配置则为 None（回退为容器内 git clone）。
        clone_timeout: ``git clone`` 超时（env ``CODEAGENTRL_CLONE_TIMEOUT``，秒）。
        exec_timeout: 普通 ``docker exec`` 命令超时（env ``CODEAGENTRL_EXEC_TIMEOUT``，秒）。
        setup_timeout: 容器内 apt/环境初始化超时（env ``CODEAGENTRL_SETUP_TIMEOUT``，秒）。
        pull_timeout: ``docker run``/镜像拉取超时（env ``CODEAGENTRL_PULL_TIMEOUT``，秒）。
        name_prefix: 容器名前缀（env ``CODEAGENTRL_NAME_PREFIX``）。
    """

    image: str = field(default_factory=lambda: os.getenv("CODEAGENTRL_IMAGE", "python:3.11-slim"))
    workdir: str = field(default_factory=lambda: os.getenv("CODEAGENTRL_WORKDIR", "/repo"))
    user: Optional[str] = field(
        default_factory=lambda: os.getenv("CODEAGENTRL_USER") or None
    )
    docker_executable: str = field(
        default_factory=lambda: os.getenv("MSWEA_DOCKER_EXECUTABLE", "docker")
    )
    install_ripgrep: bool = field(
        default_factory=lambda: _env_bool("CODEAGENTRL_INSTALL_RIPGREP", True)
    )
    apt_force_ipv4: bool = field(
        default_factory=lambda: _env_bool("CODEAGENTRL_APT_FORCE_IPV4", True)
    )
    """apt 是否强制走 IPv4（env ``CODEAGENTRL_APT_FORCE_IPV4``）。

    部分主机 IPv6 路由不可达但 DNS 优先返回 AAAA 记录，导致容器内
    ``apt-get update`` 长时间挂起；默认强制 IPv4 并加 http 超时规避。
    """
    repo_cache_dir: Optional[str] = field(
        default_factory=lambda: os.getenv("CODEAGENTRL_REPO_CACHE_DIR")
    )
    clone_timeout: int = field(
        default_factory=lambda: _env_int("CODEAGENTRL_CLONE_TIMEOUT", 600)
    )
    exec_timeout: int = field(
        default_factory=lambda: _env_int("CODEAGENTRL_EXEC_TIMEOUT", 120)
    )
    setup_timeout: int = field(
        default_factory=lambda: _env_int("CODEAGENTRL_SETUP_TIMEOUT", 600)
    )
    pull_timeout: int = field(
        default_factory=lambda: _env_int("CODEAGENTRL_PULL_TIMEOUT", 300)
    )
    name_prefix: str = field(
        default_factory=lambda: os.getenv("CODEAGENTRL_NAME_PREFIX", "codeagentrl")
    )


class EnvManager:
    """Docker 容器工厂：**每次 ``get_env`` 都新建容器，不复用、无缓存**。

    只负责两件事：**容器创建与环境准备**（镜像启动、工具安装、仓库放入 + 可选
    checkout/patch）与**销毁**（``release_env`` 幂等删除）。并发安全：每次创建
    的容器名含 uuid，天然互不冲突，无需 per-key 锁；同时创建的数量由调用方
    （``mcts.tasks.EnvFactory`` 的创建信号量）控制。
    """

    def __init__(self, config: Optional[EnvConfig] = None):
        self.config = config or EnvConfig()
        # 仅用于串行化宿主机 zip 缓存补齐（clone + 打 zip），防同路径并发重复克隆
        self._zip_lock = threading.Lock()

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------

    def get_env(
        self,
        repo_name: str,
        commit: Optional[str] = None,
        *,
        patch: Optional[str] = None,
    ) -> str:
        """**每次调用都创建一个新容器**并完成环境准备，返回容器名。

        - 仓库放入容器：zip 缓存命中则 docker cp + 解压；未命中且配置了缓存目录
          则先在宿主机 clone 打成 zip 入库再走同一路径；未配置缓存目录则容器内
          git clone；
        - ``commit`` 非空时 ``git checkout`` 切换（None 保持 HEAD，适配 SWE-Smith
          快照仓库）；``patch`` 非空时 ``git apply`` 应用（bug 引入 diff）。
        - 创建失败自动清理半成品容器并抛出异常。

        Args:
            repo_name: ``owner/repo`` 格式的仓库名。
            commit: 要切换到的 commit（SHA / 分支 / tag）；``None`` 表示保持 HEAD。
            patch: 可选的 git diff 文本，克隆后通过 ``git apply`` 应用。

        Returns:
            就绪容器的名字（外部执行完后调用 :meth:`release_env` 删除）。
        """
        repo_name = _normalize_repo(repo_name)
        return self._create_env(repo_name, commit, patch)

    def release_env(self, container_name: str) -> None:
        """销毁容器：``docker rm -f``（幂等，对空值 / 未知名安全）。"""
        if not container_name:
            return
        docker = self.config.docker_executable
        try:
            result = subprocess.run(
                [docker, "rm", "-f", container_name],
                capture_output=True, text=True,
                timeout=self.config.exec_timeout, check=False,
            )
            if result.returncode == 0:
                logger.info("Removed environment container %s", container_name)
            else:
                logger.warning(
                    "Failed to remove container %s: %s",
                    container_name, result.stderr.strip(),
                )
        except Exception as e:  # pragma: no cover - best effort cleanup
            logger.warning("Failed to remove container %s: %s", container_name, e)

    # ------------------------------------------------------------------
    # 容器创建
    # ------------------------------------------------------------------

    def _create_env(
        self, repo_name: str, commit: Optional[str], patch: Optional[str]
    ) -> str:
        docker = self.config.docker_executable
        # 创建容器前先查本地仓库 zip 缓存（缓存目录 + 仓库名）；
        # 未命中但配置了缓存目录 -> 先在宿主机拉取仓库打成 zip 存入缓存，
        # 之后与缓存命中路径完全一致（docker cp + 解压 + checkout）。
        zip_path = self._find_repo_zip(repo_name)
        if zip_path is None:
            zip_path = self._ensure_repo_zip(repo_name)
        if zip_path is not None:
            logger.info(
                "Repo zip ready for %s: %s (docker cp + unzip)", repo_name, zip_path,
            )
        container = self._new_container_name(repo_name, commit)
        logger.info(
            "Creating environment container %s for %s@%s (image=%s)",
            container, repo_name, commit, self.config.image,
        )
        try:
            run_cmd = [
                docker, "run", "-d",
                "--name", container,
                "--workdir", self.config.workdir,
            ]
            if self.config.user:
                run_cmd += ["--user", self.config.user]
            run_cmd += [self.config.image, "sleep", "infinity"]
            self._run(run_cmd, timeout=self.config.pull_timeout)
            self._ensure_tools(container)
            self._setup_repo(container, repo_name, commit, patch, zip_path)
            return container
        except Exception:
            # 创建失败时尽力清理半成品容器
            subprocess.run(
                [docker, "rm", "-f", container],
                capture_output=True, text=True, timeout=30, check=False,
            )
            raise

    def _ensure_repo_zip(self, repo_name: str) -> Optional[str]:
        """缓存未命中时补齐 zip：配置了缓存目录 -> 宿主机 clone + zip 入库并返回路径；
        未配置 -> 返回 None（走容器内 git clone 回退路径）。"""
        if not self.config.repo_cache_dir:
            return None
        zip_path = self.populate_repo_cache(repo_name)  # commit=None -> 打包 HEAD（含完整历史）
        logger.info("Repo zip cache populated for %s: %s", repo_name, zip_path)
        return zip_path

    def _new_container_name(self, repo_name: str, commit: Optional[str]) -> str:
        repo_part = _sanitize(repo_name)
        commit_part = _sanitize(commit[:8]) if commit else "head"
        return (
            f"{self.config.name_prefix}-{repo_part}-{commit_part}-"
            f"{uuid.uuid4().hex[:6]}"
        )

    def _ensure_tools(self, container: str) -> None:
        """确保容器内具备 git / unzip（+ 可选 ripgrep），缺失时走 apt 安装。"""
        missing: list[str] = []
        if not self._has_cmd(container, "git"):
            missing += ["git", "ca-certificates"]
        if not self._has_cmd(container, "unzip"):
            missing.append("unzip")
        if self.config.install_ripgrep and not self._has_cmd(container, "rg"):
            missing.append("ripgrep")
        if missing:
            pkgs = " ".join(missing)
            logger.info("Installing in %s: %s", container, pkgs)
            self._exec(
                container,
                f"apt-get {self._apt_opts()} update -qq && "
                f"DEBIAN_FRONTEND=noninteractive apt-get {self._apt_opts()} install -y -qq {pkgs}",
                timeout=self.config.setup_timeout,
            )

    def _apt_opts(self) -> str:
        """apt 额外选项：默认强制 IPv4 并限制 http 超时（见 EnvConfig.apt_force_ipv4）。"""
        if self.config.apt_force_ipv4:
            return "-o Acquire::ForceIPv4=true -o Acquire::http::Timeout=20"
        return ""

    # ------------------------------------------------------------------
    # 仓库准备：优先本地 zip 缓存，否则容器内 git clone
    # ------------------------------------------------------------------

    def _setup_repo(
        self,
        container: str,
        repo_name: str,
        commit: Optional[str],
        patch: Optional[str],
        zip_path: Optional[str],
    ) -> None:
        """把仓库放进容器工作目录：zip 缓存命中/已补齐 -> docker cp + 解压；
        未配置缓存目录 -> 容器内 git clone。

        两条路径完成后都会执行 ``git checkout <commit>``（commit 非空时）与
        可选的 ``git apply`` patch，保证最终状态与 codescout 一致。
        """
        workdir = self.config.workdir
        if zip_path is not None:
            self._copy_and_unzip(container, zip_path, repo_name)
        else:
            self._clone_into_container(container, repo_name)
        if commit:
            logger.info("Checking out %s@%s in %s", repo_name, commit, container)
            self._exec(
                container,
                f"git -C {workdir} checkout {commit}",
                timeout=self.config.exec_timeout,
            )
        if patch:
            logger.info("Applying patch in %s (repo %s)", container, repo_name)
            self._exec(
                container,
                f"git -C {workdir} apply -",
                input_text=patch,
                timeout=self.config.exec_timeout,
            )

    def _find_repo_zip(self, repo_name: str) -> Optional[str]:
        """在缓存目录中查找 ``<repo 名（/ 替换为 __）>.zip``，不存在返回 None。"""
        cache_dir = self.config.repo_cache_dir
        if not cache_dir:
            return None
        path = os.path.join(cache_dir, f"{_sanitize(repo_name)}.zip")
        return path if os.path.isfile(path) else None

    def _copy_and_unzip(self, container: str, zip_path: str, repo_name: str) -> None:
        """docker cp zip 进容器并解压到工作目录。

        zip 内可以是仓库根目录内容，也可以是一个包含仓库的顶层目录（含
        ``.git``）。解压后自动定位到含 ``.git`` 的根并移动到工作目录。
        """
        docker = self.config.docker_executable
        token = uuid.uuid4().hex[:8]
        zip_name = f"repo_{token}.zip"
        extract_dir = f"/tmp/extract_{token}"
        self._run(
            [docker, "cp", str(zip_path), f"{container}:/tmp/{zip_name}"],
            timeout=self.config.exec_timeout,
        )
        workdir = self.config.workdir
        script = (
            f"set -e; rm -rf {extract_dir}; mkdir -p {extract_dir}; "
            f"unzip -q /tmp/{zip_name} -d {extract_dir}; "
            f"if [ -d {extract_dir}/.git ]; then root={extract_dir}; "
            f"elif [ \"$(ls -A {extract_dir} | wc -l)\" = \"1\" ] && "
            f"[ -d \"{extract_dir}/$(ls {extract_dir})/.git\" ]; then "
            f"root=\"{extract_dir}/$(ls {extract_dir})\"; "
            f"else echo 'invalid repo zip: no .git found inside'; exit 1; fi; "
            f"rm -rf {workdir}; mv \"$root\" {workdir}; "
            f"rm -rf {extract_dir} /tmp/{zip_name}; "
            f"echo 'repo restored from zip: {repo_name}'"
        )
        self._exec(container, script, timeout=self.config.exec_timeout)

    def _clone_into_container(self, container: str, repo_name: str) -> None:
        """容器内 ``git clone``（与 codescout 的 clone_instance 行为一致）。

        仅在未配置 ``repo_cache_dir``（zip 缓存禁用）时作为回退路径使用。
        """
        url = f"https://github.com/{repo_name}.git"
        workdir = self.config.workdir
        # 清空工作目录（docker run --workdir 会创建该目录），随后 clone 到其中
        self._exec(
            container,
            f"rm -rf {workdir}/* {workdir}/.[!.]* 2>/dev/null || true",
            timeout=self.config.exec_timeout, check=False,
        )
        try:
            self._exec(
                container,
                f"git clone --progress {url} {workdir}",
                timeout=self.config.clone_timeout,
            )
        except RuntimeError:
            # 基础镜像里 git 存在但缺 ca-certificates 时 https clone 会 SSL 失败，
            # 安装证书后重试一次。
            logger.warning(
                "First clone attempt failed in %s; installing ca-certificates and retrying",
                container,
            )
            self._exec(
                container,
                f"apt-get {self._apt_opts()} update -qq && "
                f"DEBIAN_FRONTEND=noninteractive apt-get {self._apt_opts()} install -y -qq ca-certificates",
                timeout=self.config.setup_timeout,
            )
            self._exec(
                container,
                f"git clone --progress {url} {workdir}",
                timeout=self.config.clone_timeout,
            )

    # ------------------------------------------------------------------
    # docker 执行辅助
    # ------------------------------------------------------------------

    def _run(self, cmd: list, *, timeout: int) -> subprocess.CompletedProcess:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"docker command failed ({result.returncode}): {' '.join(cmd)}\n"
                f"{result.stderr[-2000:]}"
            )
        return result

    def _exec(
        self,
        container: str,
        command: str,
        *,
        timeout: Optional[int] = None,
        check: bool = True,
        input_text: Optional[str] = None,
    ) -> subprocess.CompletedProcess:
        docker = self.config.docker_executable
        cmd = [docker, "exec"]
        if input_text is not None:
            cmd.append("-i")
        cmd += ["--workdir", self.config.workdir, container, "bash", "-lc", command]
        result = subprocess.run(
            cmd,
            capture_output=True, text=True,
            input=input_text,
            timeout=timeout or self.config.exec_timeout,
            check=False,
        )
        if check and result.returncode != 0:
            raise RuntimeError(
                f"docker exec failed ({result.returncode}) in {container}: "
                f"{command[:200]}\n{result.stderr[-2000:]}"
            )
        return result

    def _has_cmd(self, container: str, cmd: str) -> bool:
        try:
            result = self._exec(
                container, f"command -v {cmd} >/dev/null 2>&1",
                timeout=30, check=False,
            )
            return result.returncode == 0
        except Exception:  # pragma: no cover
            return False

    # ------------------------------------------------------------------
    # zip 缓存生成（可选：提前把仓库完整克隆打成 zip 存入缓存目录）
    # ------------------------------------------------------------------

    def populate_repo_cache(
        self, repo_name: str, commit: Optional[str] = None
    ) -> str:
        """把仓库的完整 git 克隆（含 ``.git``）打成 zip 存入缓存目录。

        之后 ``get_env`` 对该仓库会命中 zip 缓存，走 ``docker cp`` + 解压，
        跳过容器内 git clone。zip 文件名 = ``<repo 名（/ 替换为 __）>.zip``。
        缓存未命中时 ``get_env`` 会自动调用本方法补齐（宿主机 clone 用
        ``_zip_lock`` 串行化，防同路径并发重复克隆）。

        Args:
            repo_name: ``owner/repo``。
            commit: 打包前 checkout 的 commit（默认 HEAD，保留完整历史，
                供后续任意 commit checkout 使用）。

        Returns:
            生成的 zip 文件路径。

        Raises:
            RuntimeError: 未配置 ``repo_cache_dir``。
        """
        repo_name = _normalize_repo(repo_name)
        cache_dir = self.config.repo_cache_dir
        if not cache_dir:
            raise RuntimeError(
                "repo_cache_dir is not configured: set EnvConfig.repo_cache_dir or "
                "the CODEAGENTRL_REPO_CACHE_DIR environment variable"
            )
        os.makedirs(cache_dir, exist_ok=True)
        zip_path = os.path.join(cache_dir, f"{_sanitize(repo_name)}.zip")
        with self._zip_lock:
            if os.path.isfile(zip_path):
                return zip_path
            tmp_dir = tempfile.mkdtemp(prefix="codeagentrl_repozip_")
            tmp_zip = f"{zip_path}.tmp-{uuid.uuid4().hex[:6]}"
            try:
                clone_dir = os.path.join(tmp_dir, _sanitize(repo_name))
                logger.info("Cloning %s on host to build zip cache", repo_name)
                subprocess.run(
                    ["git", "clone", f"https://github.com/{repo_name}.git", clone_dir],
                    check=True, capture_output=True, text=True,
                    timeout=self.config.clone_timeout,
                )
                if commit:
                    subprocess.run(
                        ["git", "-C", clone_dir, "checkout", commit],
                        check=True, capture_output=True, text=True,
                        timeout=self.config.exec_timeout,
                    )
                with zipfile.ZipFile(tmp_zip, "w", zipfile.ZIP_DEFLATED) as zf:
                    for root, _dirs, files in os.walk(clone_dir):
                        for fn in files:
                            full = os.path.join(root, fn)
                            zf.write(full, os.path.relpath(full, clone_dir))
                os.replace(tmp_zip, zip_path)  # 原子发布，避免半成品 zip 被读到
                logger.info("Populated repo cache: %s", zip_path)
                return zip_path
            finally:
                if os.path.exists(tmp_zip):
                    os.remove(tmp_zip)
                shutil.rmtree(tmp_dir, ignore_errors=True)


# ----------------------------------------------------------------------
# 模块级默认管理器（无缓存：每次 get_env 都是新建；延迟创建，确保 .env
# 的环境变量在首次使用时生效）
# ----------------------------------------------------------------------

_default_manager: Optional[EnvManager] = None


def get_default_manager() -> EnvManager:
    """按需创建并返回模块级默认 EnvManager。"""
    global _default_manager
    if _default_manager is None:
        _default_manager = EnvManager()
    return _default_manager


def get_env(
    repo_name: str,
    commit: Optional[str] = None,
    *,
    patch: Optional[str] = None,
    manager: Optional[EnvManager] = None,
) -> str:
    """获取一个**新创建**的执行环境容器名（每次调用都新建，不复用）。"""
    if manager is not None:
        return manager.get_env(repo_name, commit, patch=patch)
    return get_default_manager().get_env(repo_name, commit, patch=patch)


def release_env(container_name: str, manager: Optional[EnvManager] = None) -> None:
    """销毁容器（docker rm -f，幂等）；外部 Agent 执行完成后主动调用。"""
    if manager is not None:
        manager.release_env(container_name)
        return
    get_default_manager().release_env(container_name)


# ----------------------------------------------------------------------
# 命令行入口（便于手工验证 / 集成脚本调用）
# ----------------------------------------------------------------------

def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Manage Docker environments (create fresh / release / zip cache).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_get = sub.add_parser("get", help="create a fresh environment; print container name")
    p_get.add_argument("repo", help="owner/repo")
    p_get.add_argument("--commit", default=None, help="commit to checkout (default: HEAD)")
    p_get.add_argument("--patch-file", default=None, help="optional git diff file to apply")
    p_get.add_argument("--image", default=None, help="override base image")

    p_rel = sub.add_parser("release", help="destroy an environment container")
    p_rel.add_argument("container", help="container name returned by 'get'")

    p_cache = sub.add_parser(
        "cache", help="clone a repo on the host and store it as a zip in the repo cache"
    )
    p_cache.add_argument("repo", help="owner/repo")
    p_cache.add_argument("--commit", default=None, help="commit to checkout before zipping")
    p_cache.add_argument(
        "--cache-dir", default=None,
        help="cache dir (default: EnvConfig.repo_cache_dir / env CODEAGENTRL_REPO_CACHE_DIR)",
    )

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.cmd == "get":
        mgr = EnvManager(EnvConfig(image=args.image)) if args.image else get_default_manager()
        patch = None
        if args.patch_file:
            with open(args.patch_file, "r", encoding="utf-8") as f:
                patch = f.read()
        print(mgr.get_env(args.repo, args.commit, patch=patch))
    elif args.cmd == "release":
        get_default_manager().release_env(args.container)
    elif args.cmd == "cache":
        mgr = (
            EnvManager(EnvConfig(repo_cache_dir=args.cache_dir))
            if args.cache_dir
            else get_default_manager()
        )
        print(mgr.populate_repo_cache(args.repo, args.commit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EnvConfig",
    "EnvManager",
    "get_env",
    "get_default_manager",
    "release_env",
]
