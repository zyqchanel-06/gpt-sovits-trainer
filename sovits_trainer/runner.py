# -*- coding: utf-8 -*-
"""子进程执行器：跑一条训练命令，实时落日志，抽取进度。

不接管 stdout 语义——GPT-SoVITS 各脚本的输出格式不统一，
所以进度解析用「正则摘取 + 保留原始日志」的策略：抽得到就更新，抽不到也不影响日志完整性。
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

# 训练日志里常见的进度信号（SoVITS 的 epoch 行、tqdm 的百分比、GPT 的 step）
PROGRESS_PATTERNS = (
    re.compile(r"epoch\s*[=:]?\s*(\d+)/(\d+)", re.IGNORECASE),
    re.compile(r"Epoch:\s*(\d+)", re.IGNORECASE),
    re.compile(r"(\d{1,3})%\|"),
    re.compile(r"global_step[=:\s]+(\d+)", re.IGNORECASE),
)

ERROR_PATTERNS = (
    re.compile(r"^Traceback", re.MULTILINE),
    re.compile(r"CUDA out of memory", re.IGNORECASE),
    re.compile(r"RuntimeError:"),
    re.compile(r"No such file or directory"),
)


@dataclass
class StepResult:
    name: str
    ok: bool
    code: int
    log_path: Path
    duration_s: float = 0.0
    errors: List[str] = field(default_factory=list)

    def tail(self, lines: int = 8) -> str:
        try:
            content = self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return ""
        return "\n".join(content[-lines:])


def _match_progress(line: str) -> Optional[str]:
    for pattern in PROGRESS_PATTERNS:
        hit = pattern.search(line)
        if hit:
            return hit.group(0).strip()
    return None


def _is_error(line: str) -> bool:
    return any(pattern.search(line) for pattern in ERROR_PATTERNS)


def child_pythonpath(cwd: Path) -> str:
    """给子进程准备 PYTHONPATH。

    为什么需要：GPT-SoVITS 的脚本互相 import（slice_audio.py 要 import tools.my_utils，
    prepare_datasets 要 import GPT_SoVITS 里的模块）。官方做法是往 site-packages 写一个
    users.pth（见 webui.py），但那只对"跑过 webui 的那套环境"有效。
    这里改用 PYTHONPATH 显式传下去，不依赖任何全局状态。
    """
    cwd = Path(cwd)
    if not (cwd / "webui.py").exists():
        return ""
    candidates = [
        cwd,
        cwd / "GPT_SoVITS",
        cwd / "GPT_SoVITS" / "BigVGAN",
        cwd / "tools",
        cwd / "tools" / "asr",
        cwd / "tools" / "uvr5",
    ]
    return os.pathsep.join(str(path) for path in candidates if path.exists())


BOOTSTRAP = (
    "import os, runpy, sys\n"
    "paths, script = sys.argv[1], sys.argv[2]\n"
    "for item in reversed(paths.split(os.pathsep)):\n"
    "    if item and item not in sys.path:\n"
    "        sys.path.insert(0, item)\n"
    "sys.argv = [script] + sys.argv[3:]\n"
    "runpy.run_path(script, run_name='__main__')\n"
)


def is_embedded_python(python: str) -> bool:
    """嵌入式发行版（ComfyUI 那种 python_embeded）会忽略 PYTHONPATH。

    判据：同目录下存在 python*._pth。
    """
    try:
        path = Path(python)
        return path.exists() and any(path.parent.glob("python*._pth"))
    except OSError:
        return False


def wrap_for_embedded(cmd: List[str], injected: str) -> List[str]:
    """把 `python -s script args...` 改写成 `python -c BOOTSTRAP paths script args...`。

    这样 sys.path 由我们显式插入，不依赖 PYTHONPATH 是否被尊重。
    """
    if not injected or not cmd:
        return cmd
    rest = [str(part) for part in cmd[1:]]
    while rest and rest[0].startswith("-"):
        rest = rest[1:]  # 去掉 -s / -u 这类解释器开关
    if not rest:
        return cmd
    script, args = rest[0], rest[1:]
    return [str(cmd[0]), "-c", BOOTSTRAP, injected, script] + args


def run_step(
    name: str,
    cmd: List[str],
    cwd: Path,
    log_path: Path,
    env: Optional[Dict[str, str]] = None,
    on_progress: Optional[Callable[[str], None]] = None,
    on_line: Optional[Callable[[str], None]] = None,
    echo: bool = False,
) -> StepResult:
    """跑一条命令，边跑边写日志。

    echo=True 时同时把输出打到终端（前台跑训练时有用）。
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    merged_env = os.environ.copy()
    if env:
        merged_env.update({k: str(v) for k, v in env.items()})

    # GPT-SoVITS 的 load_audio 直接调 `ffmpeg`（依赖 PATH）。
    # 找不到就把它所在目录塞进子进程 PATH，避免 WinError 2。
    from .config import find_ffmpeg  # 局部导入：config 不依赖 runner，避免循环

    ffmpeg_path = find_ffmpeg()
    if ffmpeg_path:
        ffmpeg_dir = str(Path(ffmpeg_path).parent)
        current_path = merged_env.get("PATH", "")
        if ffmpeg_dir.lower() not in current_path.lower():
            merged_env["PATH"] = ffmpeg_dir + os.pathsep + current_path

    # 自动注入 PYTHONPATH（GPT-SoVITS 的脚本靠它互相 import）
    injected = child_pythonpath(Path(cwd))
    if injected:
        existing = merged_env.get("PYTHONPATH", "")
        merged_env["PYTHONPATH"] = injected + (os.pathsep + existing if existing else "")
        # 嵌入式 python 不认 PYTHONPATH，改用 -c bootstrap 显式注入
        if is_embedded_python(cmd[0]):
            cmd = wrap_for_embedded(cmd, injected)

    started = time.time()
    errors: List[str] = []
    header = "$ " + " ".join(str(part) for part in cmd)
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        log.write(header + "\n")
        log.write("cwd: %s\n\n" % cwd)
        log.flush()
        try:
            proc = subprocess.Popen(
                [str(part) for part in cmd],
                cwd=str(cwd),
                env=merged_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        except OSError as exc:
            log.write("启动失败: %s\n" % exc)
            return StepResult(name, False, -1, log_path, time.time() - started, [str(exc)])

        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip("\n")
            log.write(line + "\n")
            log.flush()
            if echo:
                print(line, flush=True)
            if _is_error(line):
                errors.append(line.strip())
            progress = _match_progress(line)
            if progress and on_progress:
                on_progress(progress)
            if on_line:
                on_line(line)
        code = proc.wait()

    duration = time.time() - started
    ok = code == 0
    if not ok and not errors:
        # 非零退出但没匹配到错误关键字：把日志尾部当错误带出去
        tail = StepResult(name, ok, code, log_path, duration).tail(6)
        errors = [line for line in tail.splitlines() if line.strip()][-3:]
    return StepResult(name, ok, code, log_path, duration, errors)
