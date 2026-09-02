# SPDX-License-Identifier: BSD-3-Clause
#
# CodeAgentRL Agent 通用镜像（agent/init_env.py + agent/base_agent.py + mini-swe-agent）
#
# 构建（无需安装任何额外依赖，仅需 docker）：
#     docker build -t codeagentrl-agent:ubuntu24 -f agent/Dockerfile .
#
# 使用：
#     CODEAGENTRL_IMAGE=codeagentrl-agent:ubuntu24 python scripts/demo.py owner/repo ...
#
# ---------------------------------------------------------------------------
# 为什么需要这份镜像 / Agent 对环境的需求
# ---------------------------------------------------------------------------
# 1. Shell：必须是 bash。所有动作都以 ``docker exec -w /repo <容器> bash -lc <命令>``
#    执行（init_env._exec、mini-swe-agent DockerEnvironment、ShellTool 三处一致）。
# 2. VCS：git。init_env 创建容器时在容器内 clone/checkout 仓库、应用 patch；
#    agent 任务中也要用 git diff/log 等。
# 3. 解压：unzip（+ zip）。init_env 的仓库 zip 缓存路径用 ``docker cp`` 进容器后
#    靠容器内 unzip 解压；shell 工具只读白名单也允许 tar/gzip/xz/zipinfo。
# 4. 搜索：ripgrep（rg）。codescout 风格 agent 的核心代码搜索工具
#    （init_env 默认 CODEAGENTRL_INSTALL_RIPGREP=true）。
# 5. 网络：ca-certificates、curl、wget。https git clone 需要 CA 证书
#    （缺证书时 init_env 会回退 apt 补装，预装可避免）；agent 任务常需下载依赖。
# 6. JSON/YAML：jq、yq（shell 工具白名单内的常见数据解析工具）。
# 7. 文本处理：sed/awk/grep/find/head/tail 等（Ubuntu 基础自带，白名单要求）。
# 8. Python：python3 + pip + venv + build-essential。绝大多数目标任务（SWE-bench
#    等）是 Python 项目，需要解释器、包管理、编译工具链（gcc/make）来跑测试。
# 9. 分析：file、binutils（nm/objdump/readelf/strings/ldd）、procps（ps）
#    （shell 工具只读白名单列出的系统分析命令）。
# 10. 其他语言：nodejs（部分 JS/TS 仓库需要）。
# 11. 环境约定（与 mini-swe-agent 配置对齐）：
#     - PAGER/MANPAGER=cat、LESS=-R：避免命令卡在交互式分页器
#     - PIP_PROGRESS_BAR=off、TQDM_DISABLE=1：避免进度条刷屏
#     - BASH_ENV=/root/.bashrc：``bash -c`` 非登录 shell 不读 ~/.bashrc，
#       通过 BASH_ENV 让它生效（swebench.yaml 也依赖这一约定）
#     - 工作目录 /repo（init_env 默认 CODEAGENTRL_WORKDIR）
#     - 默认以 root 运行（init_env 未配置 CODEAGENTRL_USER）
# 12. 关键收益：init_env._ensure_tools 对 git/unzip/rg 做 ``command -v`` 探测，
#     本镜像全部预装后创建容器**跳过 apt** —— 这既加快容器创建，也规避了
#     A800 这类主机上 Debian 系镜像 apt 挂起的问题（选 Ubuntu 基础镜像的原因）。
#
# 注意：容器只承载"被 Agent 操作的目标仓库环境"，Agent 本体（python 包）运行在
# 宿主机，因此本镜像不需要安装 litellm/mini-swe-agent 等 python 依赖。
# ---------------------------------------------------------------------------

# 选 Ubuntu 24.04：apt 源稳定、工具全；A800 实测 Debian(python:slim) 容器内
# apt 会挂起而 Ubuntu 正常（IPv6 路由问题，见 init_env 的 apt_force_ipv4）。
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    # 与 mini-swe-agent 环境约定对齐（见上）
    PAGER=cat \
    MANPAGER=cat \
    LESS=-R \
    PIP_PROGRESS_BAR=off \
    TQDM_DISABLE=1 \
    PYTHONUNBUFFERED=1 \
    BASH_ENV=/root/.bashrc

RUN apt-get update && apt-get install -y --no-install-recommends \
        # VCS / 证书 / 网络
        git \
        ca-certificates \
        curl \
        wget \
        # 解压 / 归档（init_env zip 缓存路径依赖 unzip）
        unzip \
        zip \
        # 代码搜索
        ripgrep \
        # JSON / YAML 处理
        jq \
        yq \
        # 系统分析（shell 工具只读白名单）
        file \
        binutils \
        procps \
        # Python 运行与构建（目标项目多为 Python）
        python3 \
        python3-pip \
        python3-venv \
        build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && printf '%s\n' \
        '# CodeAgentRL agent image' \
        "alias ll='ls -la'" \
        'export PIP_DISABLE_PIP_VERSION_CHECK=1' \
        > /root/.bashrc

# 仓库工作目录：init_env 默认 CODEAGENTRL_WORKDIR=/repo（docker run --workdir
# 会自动创建该目录，这里显式声明以保持文档一致）
WORKDIR /repo

# 自检：关键工具必须存在 —— init_env._ensure_tools 探测的是这些命令，
# 全部命中即可跳过容器创建时的 apt 安装。
RUN bash -lc 'for c in git unzip zip rg python3 pip3 jq curl wget file; do \
    command -v "$c" >/dev/null || { echo "missing required tool: $c"; exit 1; }; done'

# 与 init_env 的启动方式一致（docker run ... <image> sleep infinity），
# 手动运行镜像时也保持"常驻容器"语义。
CMD ["sleep", "infinity"]
