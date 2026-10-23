# -*- coding: utf-8 -*-
"""阶段 5：训练 GPT（语义/韵律分支，产出 .ckpt）。

与 s2 同样的思路：读 s1longer-v2.yaml 模板 → 改字段 → 写临时 yaml →
s1_train.py --config_file <临时文件>。
注意这一步还依赖两个环境变量：_CUDA_VISIBLE_DEVICES 和 hz=25hz（upstream 约定）。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from ..config import TrainerConfig
from ..runner import StepResult, run_step
from ..state import Task

PREP_MODEL = "gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"
TEMPLATES = {
    "v1": "GPT_SoVITS/configs/s1longer.yaml",
    "v2": "GPT_SoVITS/configs/s1longer-v2.yaml",
}


def template_path(config: TrainerConfig) -> Path:
    rel = TEMPLATES.get(config.version, TEMPLATES["v2"])
    return config.gpt_sovits_root / rel


def default_pretrained(config: TrainerConfig) -> str:
    primary = config.gpt_sovits_root / "GPT_SoVITS" / "pretrained_models" / PREP_MODEL
    if primary.exists():
        return str(primary)
    # 有的发行版把权重平铺在 pretrained_models 根下
    flat = config.gpt_sovits_root / "GPT_SoVITS" / "pretrained_models" / Path(PREP_MODEL).name
    return str(flat)


def build_config(
    config: TrainerConfig,
    task: Task,
    *,
    batch_size: int = 4,
    epochs: int = 8,
    save_every_epoch: int = 4,
    pretrained_s1: Optional[str] = None,
    if_dpo: bool = False,
) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("训练 GPT 需要 PyYAML：pip install pyyaml") from exc

    template = template_path(config)
    if not template.exists():
        raise FileNotFoundError("GPT 训练配置模板不存在：%s" % template)
    data = yaml.safe_load(template.read_text(encoding="utf-8"))

    exp_dir = config.stage_dir(task.voice, "prep")
    data["train"]["batch_size"] = batch_size
    data["train"]["epochs"] = epochs
    data["train"]["save_every_n_epoch"] = save_every_epoch
    data["train"]["if_save_every_weights"] = True
    data["train"]["if_save_latest"] = True
    data["train"]["if_dpo"] = if_dpo
    data["train"]["half_weights_save_dir"] = str(config.weights_dir)
    data["train"]["exp_name"] = task.voice
    if not config.is_half:
        data["train"]["precision"] = "32"
        data["train"]["batch_size"] = max(1, batch_size // 2)
    data["pretrained_s1"] = pretrained_s1 or default_pretrained(config)
    data["train_semantic_path"] = str(exp_dir / "6-name2semantic.tsv")
    data["train_phoneme_path"] = str(exp_dir / "2-name2text.txt")
    data["output_dir"] = str(exp_dir / ("logs_s1_" + config.version))
    return data


def write_config(config: TrainerConfig, task: Task, data: Dict[str, Any]) -> Path:
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("训练 GPT 需要 PyYAML：pip install pyyaml") from exc
    target = config.stage_dir(task.voice, "s1") / "tmp_s1.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return target


def run(
    config: TrainerConfig,
    task: Task,
    *,
    batch_size: int = 4,
    epochs: int = 8,
    save_every_epoch: int = 4,
    pretrained_s1: Optional[str] = None,
    if_dpo: bool = False,
) -> StepResult:
    # 先验证上游产物在不在（s1 依赖 prep 阶段的两个文件）
    exp_dir = config.stage_dir(task.voice, "prep")
    for required in ("6-name2semantic.tsv", "2-name2text.txt"):
        if not (exp_dir / required).exists():
            raise FileNotFoundError(
                "缺少 %s —— 请先跑完 prep 阶段（特征提取）" % required
            )

    data = build_config(
        config, task,
        batch_size=batch_size, epochs=epochs, save_every_epoch=save_every_epoch,
        pretrained_s1=pretrained_s1, if_dpo=if_dpo,
    )
    tmp_config = write_config(config, task, data)
    return run_step(
        "s1",
        [config.python, "-s", "GPT_SoVITS/s1_train.py", "--config_file", str(tmp_config)],
        cwd=config.gpt_sovits_root,
        log_path=config.log_path(task.voice, "s1"),
        env={"_CUDA_VISIBLE_DEVICES": config.cuda_visible, "hz": "25hz"},
    )
