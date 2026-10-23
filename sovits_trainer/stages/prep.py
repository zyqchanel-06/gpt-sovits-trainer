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
    """三个脚本共用这一套环境变量。

    注意这三个脚本读的变量名不完全一样（都是上游约定）：
      1-get-text.py     需要 bert_pretrained_dir
      2-get-hubert-*.py 需要 cnhubert_base_dir
      3-get-semantic.py 需要 pretrained_s2G + s2config_path
    少一个就 AttributeError / TypeError，所以统一在这里备齐。
    """
    from . import s2 as s2_stage  # 局部导入：复用 s2 的模板/权重默认值

    pretrained = config.gpt_sovits_root / "GPT_SoVITS" / "pretrained_models"
    s2_defaults = s2_stage.default_pretrained(config)
    return {
        "inp_text": str(list_path),
        "inp_wav_dir": str(sliced_dir),
        "exp_name": task.voice,
        "i_part": "0",
        "all_parts": "1",
        "opt_dir": str(exp_dir),
        # 1-get-text
        "bert_pretrained_dir": str(pretrained / "chinese-roberta-wwm-ext-large"),
        # 2-get-hubert
        "cnhubert_base_dir": str(pretrained / "chinese-hubert-base"),
        # 3-get-semantic
        "pretrained_s2G": s2_defaults["pretrained_s2G"],
        "s2config_path": str(s2_stage.template_path(config)),
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

    if all(item.ok for item in results):
        normalize_artifacts(exp_dir)
    return results


# 上游产物带分片后缀（-0），但训练脚本按无后缀的文件名读。
# 单卡训练时补一份无后缀的别名，避免 s2/s1 直接 assert 失败。
SINGLE_ALIASES = (
    ("2-name2text.txt", "2-name2text-*.txt"),
    ("6-name2semantic.tsv", "6-name2semantic-*.tsv"),
)


def normalize_artifacts(exp_dir: Path) -> List[str]:
    """给单分片产物做无后缀别名，返回实际创建的文件名。"""
    import shutil

    created: List[str] = []
    for alias, pattern in SINGLE_ALIASES:
        target = exp_dir / alias
        if target.exists():
            continue
        hits = sorted(exp_dir.glob(pattern))
        if len(hits) == 1:
            shutil.copy2(hits[0], target)
            created.append(alias)
    return created


def _first_match(folder: Path, pattern: str) -> Optional[Path]:
    """上游产物会带分片后缀（2-name2text-0.txt），用 glob 兼容单机/多卡两种情况。"""
    if not folder.exists():
        return None
    hits = sorted(folder.glob(pattern))
    return hits[0] if hits else None


def artifacts(config: TrainerConfig, task: Task) -> Dict[str, Optional[str]]:
    """检查三个关键中间产物是否齐了。"""
    exp_dir = config.stage_dir(task.voice, "prep")
    cnhubert = exp_dir / "4-cnhubert"
    found = {
        "name2text": _first_match(exp_dir, "2-name2text.txt") or _first_match(exp_dir, "2-name2text*.txt"),
        "cnhubert": cnhubert if cnhubert.is_dir() and any(cnhubert.iterdir()) else None,
        "semantic": _first_match(exp_dir, "6-name2semantic*.tsv"),
    }
    return {key: (str(path) if path else None) for key, path in found.items()}
