# -*- coding: utf-8 -*-
"""阶段 2：语音识别标注（可跳过）。

两种模式：
  - 自动：调 GPT-SoVITS 自带的 faster-whisper / FunASR 生成 .list
  - 跳过：用户自己提供一份 .list（格式：<wav路径>|<说话人>|<语言>|<文本>）

自动识别对小语种/专业词容易错，所以跑完最好人工校对一遍再来训练。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import TrainerConfig
from ..runner import StepResult, run_step
from ..state import Task

# GPT-SoVITS 自带的两个 ASR 后端（tools/asr/）
ASR_BACKENDS = {
    "faster-whisper": "tools/asr/fasterwhisper_asr.py",
    "funasr": "tools/asr/funasr_asr.py",
}


def build_command(
    config: TrainerConfig,
    input_dir: Path,
    output_dir: Path,
    *,
    backend: str = "faster-whisper",
    model_size: str = "large-v3",
    language: str = "zh",
    precision: str = "float16",
) -> List[str]:
    if backend not in ASR_BACKENDS:
        raise ValueError("未知 ASR 后端：%s（可选：%s）" % (backend, ", ".join(ASR_BACKENDS)))
    return [
        config.python, "-s", ASR_BACKENDS[backend],
        "-i", str(input_dir),
        "-o", str(output_dir),
        "-s", model_size,
        "-l", language,
        "-p", precision,
    ]


def find_list(folder: Path) -> Optional[Path]:
    """ASR 产物是一个 .list 文件，名字随输入目录变，所以这里扫一下。"""
    if not folder.exists():
        return None
    lists = sorted(folder.glob("*.list"), key=lambda p: p.stat().st_mtime, reverse=True)
    return lists[0] if lists else None


def run(
    config: TrainerConfig,
    task: Task,
    *,
    input_dir: Optional[Path] = None,
    provided_list: Optional[Path] = None,
    backend: str = "faster-whisper",
    model_size: str = "large-v3",
    language: str = "zh",
    precision: str = "float16",
) -> Dict[str, Any]:
    """返回 {skipped, list_path, result?}。"""
    # 用户自己给了标注 → 直接用，不跑 ASR
    if provided_list is not None:
        provided = Path(provided_list).expanduser().resolve()
        if not provided.exists():
            raise FileNotFoundError("标注文件不存在：%s" % provided)
        return {"skipped": True, "list_path": str(provided), "reason": "使用用户提供的标注"}

    source = Path(input_dir) if input_dir else (config.stage_dir(task.voice, "slice") / "wav")
    if not source.exists():
        raise FileNotFoundError("切片目录不存在：%s（先跑 slice 阶段）" % source)

    output_dir = config.stage_dir(task.voice, "asr")
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_command(
        config, source, output_dir,
        backend=backend, model_size=model_size, language=language, precision=precision,
    )
    result = run_step(
        "asr",
        cmd,
        cwd=config.gpt_sovits_root,
        log_path=config.log_path(task.voice, "asr"),
    )
    found = find_list(output_dir)
    return {
        "skipped": False,
        "list_path": str(found) if found else None,
        "result": result,
        "output_dir": str(output_dir),
    }
