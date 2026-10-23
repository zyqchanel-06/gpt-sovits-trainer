# -*- coding: utf-8 -*-
"""阶段 4：训练 SoVITS（音色/音质分支，产出 .pth）。

做法与官方 webui 完全一致：读 GPT_SoVITS/configs/s2.json 模板 → 改字段 →
写临时配置 → s2_train.py --config <临时文件>。
不复制训练逻辑，所以上游升级时只要字段名没变就能继续用。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import TrainerConfig
from ..runner import StepResult, run_step
from ..state import Task

# 不同版本用不同的模板和预训练权重
VERSION_TEMPLATES = {
    "v1": ("GPT_SoVITS/configs/s2.json", "s2G488k.pth", "s2D488k.pth"),
    "v2": ("GPT_SoVITS/configs/s2.json", "gsv-v2final-pretrained/s2G2333k.pth", "gsv-v2final-pretrained/s2D2333k.pth"),
}
DEFAULT_TEMPLATE = "GPT_SoVITS/configs/s2.json"


def template_path(config: TrainerConfig) -> Path:
    rel = VERSION_TEMPLATES.get(config.version, (DEFAULT_TEMPLATE,))[0]
    return config.gpt_sovits_root / rel


def default_pretrained(config: TrainerConfig) -> Dict[str, str]:
    base = config.gpt_sovits_root / "GPT_SoVITS" / "pretrained_models"
    _, s2g, s2d = VERSION_TEMPLATES.get(config.version, ("", "gsv-v2final-pretrained/s2G2333k.pth",
                                                       "gsv-v2final-pretrained/s2D2333k.pth"))
    return {"pretrained_s2G": str(base / s2g), "pretrained_s2D": str(base / s2d)}


def build_config(
    config: TrainerConfig,
    task: Task,
    *,
    batch_size: int = 4,
    epochs: int = 8,
    save_every_epoch: int = 4,
    text_low_lr_rate: float = 0.4,
    lora_rank: int = 32,
    grad_ckpt: bool = False,
    pretrained_s2G: Optional[str] = None,
    pretrained_s2D: Optional[str] = None,
) -> Dict[str, Any]:
    """读模板并覆盖字段，返回完整配置字典。"""
    template = template_path(config)
    if not template.exists():
        raise FileNotFoundError("SoVITS 配置模板不存在：%s" % template)
    data = json.loads(template.read_text(encoding="utf-8"))

    exp_dir = config.stage_dir(task.voice, "prep")
    defaults = default_pretrained(config)

    data["train"]["batch_size"] = batch_size
    data["train"]["epochs"] = epochs
    data["train"]["text_low_lr_rate"] = text_low_lr_rate
    data["train"]["pretrained_s2G"] = pretrained_s2G or defaults["pretrained_s2G"]
    data["train"]["pretrained_s2D"] = pretrained_s2D or defaults["pretrained_s2D"]
    data["train"]["if_save_latest"] = True
    data["train"]["if_save_every_weights"] = True
    data["train"]["save_every_epoch"] = save_every_epoch
    data["train"]["gpu_numbers"] = config.cuda_visible
    data["train"]["grad_ckpt"] = grad_ckpt
    data["train"]["lora_rank"] = lora_rank
    if not config.is_half:
        data["train"]["fp16_run"] = False
        data["train"]["batch_size"] = max(1, batch_size // 2)
    data["model"]["version"] = config.version
    data["data"]["exp_dir"] = data["s2_ckpt_dir"] = str(exp_dir)
    data["save_weight_dir"] = str(config.weights_dir)
    data["name"] = task.voice
    data["version"] = config.version
    return data


def write_config(config: TrainerConfig, task: Task, data: Dict[str, Any]) -> Path:
    target = config.stage_dir(task.voice, "s2") / "tmp_s2.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def run(
    config: TrainerConfig,
    task: Task,
    *,
    batch_size: int = 4,
    epochs: int = 8,
    save_every_epoch: int = 4,
    text_low_lr_rate: float = 0.4,
    lora_rank: int = 32,
    grad_ckpt: bool = False,
    pretrained_s2G: Optional[str] = None,
    pretrained_s2D: Optional[str] = None,
) -> StepResult:
    data = build_config(
        config, task,
        batch_size=batch_size, epochs=epochs, save_every_epoch=save_every_epoch,
        text_low_lr_rate=text_low_lr_rate, lora_rank=lora_rank, grad_ckpt=grad_ckpt,
        pretrained_s2G=pretrained_s2G, pretrained_s2D=pretrained_s2D,
    )
    # 上游 webui 在开训前会 makedirs，s2_train.py 自己不创建 —— 缺了会在保存时炸
    exp_dir = config.stage_dir(task.voice, "prep")
    (exp_dir / ("logs_s2_" + config.version)).mkdir(parents=True, exist_ok=True)
    tmp_config = write_config(config, task, data)
    return run_step(
        "s2",
        [config.python, "-s", "GPT_SoVITS/s2_train.py", "--config", str(tmp_config)],
        cwd=config.gpt_sovits_root,
        log_path=config.log_path(task.voice, "s2"),
    )
