# -*- coding: utf-8 -*-
"""阶段 3：特征提取（三个脚本依次跑）。

1-get-text.py      文本 → 音素（2-name2text.txt）
2-get-hubert-wav32k.py  音频 → SSL 特征（4-cnhubert/*）
3-get-semantic.py  导出语义 token（6-name2semantic.tsv）

这三个脚本**不用命令行参数**，而是读环境变量（inp_text / inp_wav_dir / opt_dir ...），
这是 GPT-SoVITS upstream 的既定约定（见 webui.py open1a/1b/1c）。
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from ..config import TrainerConfig
from ..runner import StepResult, run_step
from ..state import Task

PREP_SCRIPTS = (
    ("text", "GPT_SoVITS/prepare_datasets/1-get-text.py"),
    ("hubert", "GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py"),
    ("semantic", "GPT_SoVITS/prepare_datasets/3-get-semantic.py"),
)


def build_env(
    config: TrainerConfig,
    task: Task,
    list_path: Path,
    sliced_dir: Path,
    exp_dir: Path,
) -> Dict[str, str]:
    """三个脚本共用这一套环境变量。"""
    return {
        "inp_text": str(list_path),
        "inp_wav_dir": str(sliced_dir),
        "exp_name": task.voice,
        "i_part": "0",
        "all_parts": "1",
        "opt_dir": str(exp_dir),
        "bert_pretrained_dir": str(
            config.gpt_sovits_root / "GPT_SoVITS" / "pretrained_models" / "chinese-roberta-wwm-ext-large"
        ),
        "_CUDA_VISIBLE_DEVICES": config.cuda_visible,
        "is_half": "True" if config.is_half else "False",
        "version": config.version,
    }


def run(
    config: TrainerConfig,
    task: Task,
    list_path: Path,
    sliced_dir: Path,
    *,
    stop_on_error: bool = True,
) -> List[StepResult]:
    """依次跑三个脚本；任何一个失败默认中断（后面的步骤依赖前面的产物）。"""
    list_path = Path(list_path).expanduser().resolve()
    sliced_dir = Path(sliced_dir).expanduser().resolve()
    if not list_path.exists():
        raise FileNotFoundError("标注文件不存在：%s" % list_path)
    if not sliced_dir.exists():
        raise FileNotFoundError("切片目录不存在：%s" % sliced_dir)

    exp_dir = config.stage_dir(task.voice, "prep")
    exp_dir.mkdir(parents=True, exist_ok=True)
    env = build_env(config, task, list_path, sliced_dir, exp_dir)

    results: List[StepResult] = []
    for name, script in PREP_SCRIPTS:
        result = run_step(
            "prep-" + name,
            [config.python, "-s", script],
            cwd=config.gpt_sovits_root,
            log_path=config.log_path(task.voice, "prep-" + name),
            env=env,
        )
        results.append(result)
        if stop_on_error and not result.ok:
            break
    return results


def artifacts(config: TrainerConfig, task: Task) -> Dict[str, Optional[str]]:
    """检查三个关键中间产物是否齐了。"""
    exp_dir = config.stage_dir(task.voice, "prep")
    expected = {
        "name2text": exp_dir / "2-name2text.txt",
        "cnhubert": exp_dir / "4-cnhubert",
        "semantic": exp_dir / "6-name2semantic.tsv",
    }
    return {key: (str(path) if path.exists() else None) for key, path in expected.items()}
