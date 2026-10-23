# -*- coding: utf-8 -*-
"""阶段 1：音频切片。

调 tools/slice_audio.py，把长录音按静音切成 3~10 秒的小段。
切片质量直接决定后面标注和训练的成败，所以参数要能调。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import TrainerConfig
from ..runner import StepResult, run_step
from ..state import Task


def build_command(
    config: TrainerConfig,
    audio_path: Path,
    output_dir: Path,
    *,
    threshold: float = -34,
    min_length: int = 4000,
    min_interval: int = 300,
    hop_size: int = 10,
    max_sil_kept: int = 500,
    max_length: int = 10000,
    alpha: float = 0.25,
) -> List[str]:
    """拼出 slice_audio.py 的命令行（9 个位置参数，顺序不能错）。"""
    return [
        config.python, "-s", "tools/slice_audio.py",
        str(audio_path), str(output_dir),
        str(threshold), str(min_length), str(min_interval),
        str(hop_size), str(max_sil_kept), str(max_length), str(alpha),
        "0", "1",
    ]


def run(
    config: TrainerConfig,
    task: Task,
    audio_path: Path,
    *,
    threshold: float = -34,
    min_length: int = 4000,
    min_interval: int = 300,
    max_sil_kept: int = 500,
    max_length: int = 10000,
    alpha: float = 0.25,
) -> StepResult:
    audio_path = Path(audio_path).expanduser().resolve()
    if not audio_path.exists():
        raise FileNotFoundError("音频不存在：%s" % audio_path)

    output_dir = config.stage_dir(task.voice, "slice") / "wav"
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = build_command(
        config, audio_path, output_dir,
        threshold=threshold, min_length=min_length, min_interval=min_interval,
        max_sil_kept=max_sil_kept, max_length=max_length, alpha=alpha,
    )
    return run_step(
        "slice",
        cmd,
        cwd=config.gpt_sovits_root,
        log_path=config.log_path(task.voice, "slice"),
    )


def collect_output(config: TrainerConfig, task: Task) -> Dict[str, Any]:
    """切完片统计一下：多少段、总时长。"""
    folder = config.stage_dir(task.voice, "slice") / "wav"
    files = sorted(folder.glob("*.wav")) if folder.exists() else []
    return {"dir": str(folder), "count": len(files), "files": [f.name for f in files]}
