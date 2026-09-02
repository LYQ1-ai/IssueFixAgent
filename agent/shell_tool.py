# SPDX-License-Identifier: BSD-3-Clause

"""Read-only shell tool for root-cause analysis agents.

Design follows the bash tool logic of ``ref_papers/mini-swe-agent``:

- A single ``bash`` tool exposed through an OpenAI function-calling schema.
- ``execute()`` returns ``{"output", "returncode", "exception_info"}``.
- Stateless execution: every action runs independently (no persistent shell session).

Environment management is **delegated to ``agent.init_env``**: the execution
environment (a Docker container with the target repo cloned and checked out at
``/repo``) is created and cached by :class:`agent.init_env.EnvManager` and passed
into :class:`ShellTool` at construction time. ``ShellTool`` itself never creates,
mounts, or removes containers; it only:

1. validates that the command is read-only (command-level filtering), and
2. runs it inside the given container via ``docker exec``.

Batch rollout requirements (40k instances / 128 repos / high concurrency):

- Container pooling & per-(repo, commit) caching live in ``init_env``;
  concurrent LLM requests against the same repo share the same container.
- Read-only enforcement: command-level filtering here, plus whatever the
  environment provides (repo mounted read-only etc. is the environment's job).

Agent ergonomics: the agent only ever passes the ``command`` argument; the repo
and container context are bound to the tool instance at construction time.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import threading
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("agent.shell_tool")

# ---------------------------------------------------------------------------
# Read-only command validation
# ---------------------------------------------------------------------------


class ReadOnlyViolation(Exception):
    """Raised when a command violates the read-only policy."""

    def __init__(self, reason: str, command: str):
        self.reason = reason
        self.command = command
        super().__init__(f"Command rejected (read-only policy): {reason}")


# Patterns that are NEVER allowed, regardless of tmp-write policy: package
# managers, VCS mutations, privilege/device changes, killing, system writes.
_HARD_BLOCK_PATTERNS = [
    r"\bchmod\b",
    r"\bchown\b",
    r"\bchattr\b",
    r"\bdd\b",
    r"\btruncate\b",
    r"\bmkfs\b",
    r"\bmount\b",
    r"\bumount\b",
    r"\binstall\b",
    r"\bshred\b",
    r"\bkill\b",
    r"\bpkill\b",
    r"\bkillall\b",
    r"\bshutdown\b",
    r"\breboot\b",
    r"\bsed\s+-i\b",
    r"\bperl\s+-i\b",
    r"\bapt(-get)?\s+(update|install|remove|purge|upgrade)",
    r"\byum\s+(install|remove|erase|update)",
    r"\bbrew\s+(install|uninstall|upgrade)",
    r"\bpip\s+install\b",
    r"\bpip[0-9.]*\s+install\b",
    r"\buv\s+(add|remove|sync|pip)",
    r"\bnpm\s+(install|uninstall|update|ci|init|add)",
    r"\bpnpm\s+(install|remove|update)",
    r"\byarn\s+(add|remove|install)",
    r"\bgit\s+(add|commit|push|pull|clone|init|fetch|merge|rebase|reset|"
    r"clean|stash|rm|mv|tag|branch\s+-[dD]|checkout\s+-[bB]|switch\s+-c|restore\s+--staged)",
    r"\bmake\b",
    r"\bcmake\b",
    r"\bpython[0-9.]*\s+-m\s+(pip|conda|venv|ensurepip|compileall)",
    r"\bpython[0-9.]*\s+-c\b[^\n]*\bopen\([^\n]*['\"]w['\"]",
]

_HARD_BLOCK_RE = re.compile("|".join(_HARD_BLOCK_PATTERNS), re.IGNORECASE)

# File-mutation commands: allowed only when their path arguments live under an
# allowed writable root (e.g. /tmp) when allow_tmp_writes is enabled.
_FILE_MUTATION_COMMANDS = {
    "rm", "mv", "cp", "touch", "mkdir", "rmdir", "ln", "tee", "unlink", "install",
}

# Redirection operators that create/overwrite/append files.
_REDIRECT_RE = re.compile(r"(?<![12&])(?:>>|>)\s*(\S+)")

# Optional strict mode: only these command names (first token of each
# sub-command) are allowed.
_READONLY_ALLOWLIST = {
    "ls", "cat", "head", "tail", "grep", "rg", "find", "sed", "awk", "gawk",
    "wc", "sort", "uniq", "diff", "cut", "paste", "nl", "od", "hexdump",
    "file", "stat", "du", "df", "ps", "env", "printenv", "echo", "printf",
    "pwd", "which", "type", "command", "test", "[", "true", "false", "cd",
    "export", "unset", "set", "shift", "read", "for", "while", "until",
    "if", "then", "else", "elif", "fi", "case", "esac", "do", "done",
    "python", "python3", "python3.10", "python3.11", "python3.12", "node",
    "git", "gitlog", "bash", "sh", "zsh", "timeout", "xargs", "jq", "yq",
    "base64", "tar", "unzip", "zipinfo", "zcat", "gzip", "gunzip", "xz",
    "basename", "dirname", "realpath", "readlink", "date", "cal", "nproc",
    "getconf", "locale", "ldd", "strings", "nm", "objdump", "readelf",
}

_SUBCOMMAND_SPLIT_RE = re.compile(r"[;&|\n]")


def _is_allowed_tmp_target(path: str) -> bool:
    """True if a write target is under an allowed writable tmp root.

    Paths are normalized first so ``/tmp/../repo`` is correctly rejected.
    """
    if path == "/dev/null":
        return True
    # Expand $TMPDIR / ${TMPDIR} references conservatively (only as whole prefix).
    expanded = path
    for var, val in (("$TMPDIR", "/tmp"), ("${TMPDIR}", "/tmp")):
        if expanded == var:
            expanded = val
        elif expanded.startswith(var + "/"):
            expanded = val + expanded[len(var):]
    norm = os.path.normpath(expanded)
    if norm == "/tmp":
        return True
    return norm.startswith("/tmp/") or norm.startswith("/dev/null")


def _reject(reason: str, command: str) -> None:
    raise ReadOnlyViolation(reason, command)


def validate_readonly(
    command: str,
    *,
    allowlist_only: bool = False,
    allowlist: Optional[set[str]] = None,
    allow_tmp_writes: bool = True,
) -> None:
    """Validate a shell command against the read-only policy.

    Raises :class:`ReadOnlyViolation` if the command is not read-only.

    When ``allow_tmp_writes`` is True, writes targeting the container's tmpfs
    scratch space (``/tmp``) are permitted; all other writes are rejected.
    """
    command = command.strip()
    if not command:
        _reject("empty command", command)

    # Layer 1a: hard-blocked operations (never allowed).
    if (m := _HARD_BLOCK_RE.search(command)) is not None:
        _reject(f"contains forbidden pattern {m.group(0)!r}", command)

    # Layer 1b: file-mutation commands — check their path arguments.
    for sub in _SUBCOMMAND_SPLIT_RE.split(command):
        sub = sub.strip().lstrip("(")
        if not sub:
            continue
        tokens = sub.split()
        # strip env assignments / wrappers
        while tokens and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", tokens[0]):
            tokens.pop(0)
        if tokens and tokens[0] in {"time", "command", "nohup", "sudo"}:
            tokens.pop(0)
        if not tokens:
            continue
        cmd_name = tokens[0].split("/")[-1]
        if cmd_name in _FILE_MUTATION_COMMANDS:
            paths = [
                t for t in tokens[1:]
                if not t.startswith("-") and not re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", t)
            ]
            if allow_tmp_writes:
                bad = [p for p in paths if not _is_allowed_tmp_target(p)]
                if bad:
                    _reject(
                        f"{cmd_name}: write target {bad[0]!r} outside allowed tmp space",
                        command,
                    )
            elif paths:
                _reject(f"{cmd_name}: write command not allowed", command)

    # Layer 1c: redirections — check the target.
    for m in _REDIRECT_RE.finditer(command):
        target = m.group(1).strip("\"'")
        if allow_tmp_writes and _is_allowed_tmp_target(target):
            continue
        _reject(f"redirection to {target!r} is not allowed", command)

    # Layer 2: optional strict allowlist of read-only command names.
    if allowlist_only:
        allowed = allowlist or _READONLY_ALLOWLIST
        for sub in _SUBCOMMAND_SPLIT_RE.split(command):
            sub = sub.strip().lstrip("(")
            if not sub:
                continue
            tokens = sub.split()
            while tokens and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", tokens[0]):
                tokens.pop(0)
            if tokens and tokens[0] in {"time", "command", "nohup", "sudo"}:
                tokens.pop(0)
            while tokens and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", tokens[0]):
                tokens.pop(0)
            if not tokens:
                continue
            if tokens[0].split("/")[-1] not in allowed:
                _reject(f"command {tokens[0]!r} not in read-only allowlist", command)


# ---------------------------------------------------------------------------
# Shell tool (execution only; environment is injected from init_env)
# ---------------------------------------------------------------------------


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
class ShellToolConfig:
    """Execution configuration for :class:`ShellTool`.

    所有字段都有代码内默认值，且均可用环境变量覆盖（见项目根目录
    ``.env_template``）。显式传入构造参数的优先级高于环境变量。

    Args:
        workdir: Working directory inside the container (the repo checkout
            path created by ``init_env``; env ``MSWEA_WORKDIR``, default ``/repo``).
        timeout: Per-command timeout in seconds (env ``MSWEA_TIMEOUT``).
        max_output_chars: Truncate returned output to this many characters
            (head + tail, mirroring mini-swe-agent's observation template;
            env ``MSWEA_MAX_OUTPUT_CHARS``).
        max_concurrent: Max concurrent ``docker exec`` calls against this
            container (``0`` = unlimited; env ``MSWEA_MAX_CONCURRENT``).
        docker_executable: Path to the docker CLI (env ``MSWEA_DOCKER_EXECUTABLE``).
        readonly_allowlist_only: If True, additionally restrict commands to an
            allowlist of read-only command names (strict mode;
            env ``MSWEA_READONLY_ALLOWLIST_ONLY``).
        allow_tmp_writes: If True, writes targeting ``/tmp`` are permitted
            (env ``MSWEA_ALLOW_TMP_WRITES``).
    """

    workdir: str = field(default_factory=lambda: os.getenv("MSWEA_WORKDIR", "/repo"))
    timeout: int = field(default_factory=lambda: _env_int("MSWEA_TIMEOUT", 30))
    max_output_chars: int = field(
        default_factory=lambda: _env_int("MSWEA_MAX_OUTPUT_CHARS", 10000)
    )
    max_concurrent: int = field(
        default_factory=lambda: _env_int("MSWEA_MAX_CONCURRENT", 4)
    )
    docker_executable: str = field(
        default_factory=lambda: os.getenv("MSWEA_DOCKER_EXECUTABLE", "docker")
    )
    readonly_allowlist_only: bool = field(
        default_factory=lambda: _env_bool("MSWEA_READONLY_ALLOWLIST_ONLY", False)
    )
    allow_tmp_writes: bool = field(
        default_factory=lambda: _env_bool("MSWEA_ALLOW_TMP_WRITES", True)
    )


class ShellTool:
    """A read-only ``bash`` tool bound to an execution environment.

    The execution environment (container name) is provided by
    :func:`agent.init_env.get_env` and passed in at construction time; this
    class performs no environment management of its own.

    The agent only calls ``execute(command)`` (or uses the tool via the
    function-calling schema from :meth:`get_schema`); the container context is
    bound to the instance.
    """

    def __init__(
        self,
        container: str,
        *,
        config: Optional[ShellToolConfig] = None,
    ):
        self.container = container
        self.config = config or ShellToolConfig()
        if self.config.max_concurrent > 0:
            self._semaphore = threading.BoundedSemaphore(self.config.max_concurrent)
        else:
            self._semaphore = None

    # -- tool interface -----------------------------------------------------

    def get_schema(self) -> dict:
        """Return the OpenAI function-calling schema for this tool."""
        return {
            "type": "function",
            "function": {
                "name": "bash",
                "description": (
                    "Execute a read-only bash command inside the target repository. "
                    "Use it to inspect files, search the codebase, read git history, "
                    "and run read-only analysis scripts. Write operations are "
                    "forbidden. The current working directory is the repo root."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The read-only bash command to execute.",
                        }
                    },
                    "required": ["command"],
                },
            },
        }

    def execute(self, command: str) -> dict:
        """Execute a read-only command in the environment container.

        Returns a dict with keys ``output``, ``returncode``, ``exception_info``
        (mirroring mini-swe-agent's environment contract).
        """
        validate_readonly(
            command,
            allowlist_only=self.config.readonly_allowlist_only,
            allow_tmp_writes=self.config.allow_tmp_writes,
        )
        if self._semaphore is not None:
            with self._semaphore:
                return self._run_in_container(command)
        return self._run_in_container(command)

    __call__ = execute

    # -- internals ----------------------------------------------------------

    def _run_in_container(self, command: str) -> dict:
        docker = self.config.docker_executable
        # Wrap with `timeout` inside the container so runaway processes are
        # killed even if the docker CLI itself times out.
        inner = f"timeout --signal=KILL {self.config.timeout} bash -lc {shlex_quote(command)}"
        cmd = [
            docker, "exec",
            "--workdir", self.config.workdir,
            self.container,
            "bash", "-lc", inner,
        ]
        timeout_s = self.config.timeout + 10
        try:
            result = subprocess.run(
                cmd, text=True, encoding="utf-8", errors="replace",
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                timeout=timeout_s,
            )
            return {
                "output": _truncate_output(result.stdout, self.config.max_output_chars),
                "returncode": result.returncode,
                "exception_info": "",
            }
        except subprocess.TimeoutExpired as e:
            raw = e.output or b""
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="replace")
            return {
                "output": _truncate_output(raw, self.config.max_output_chars),
                "returncode": -1,
                "exception_info": (
                    f"Command timed out after {self.config.timeout}s and was killed."
                ),
            }
        except Exception as e:
            return {
                "output": "",
                "returncode": -1,
                "exception_info": f"An error occurred while executing the command: {e}",
            }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def shlex_quote(s: str) -> str:
    """Quote a string for POSIX shell (shlex.quote does not handle newlines)."""
    return "'" + s.replace("'", "'\"'\"'") + "'"


def _truncate_output(output: str, max_chars: int) -> str:
    """Truncate output to head + tail, mirroring mini-swe-agent's template."""
    if len(output) <= max_chars or max_chars <= 0:
        return output
    half = max_chars // 2
    head = output[:half]
    tail = output[-half:]
    elided = len(output) - max_chars
    return f"{head}\n... [elided_chars: {elided}] ...\n{tail}"


__all__ = [
    "ReadOnlyViolation",
    "ShellTool",
    "ShellToolConfig",
    "validate_readonly",
]
