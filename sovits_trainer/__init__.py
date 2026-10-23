# -*- coding: utf-8 -*-
"""GPT-SoVITS 训练编排器。

    from sovits_trainer import TrainerConfig, Task, run_pipeline

    config = TrainerConfig.load()
    task = Task.load(config.work_root, "mychar")
    run_pipeline(config, task, {"audio_path": "./recordings"})

设计取向：不重写训练，只编排。所有训练动作都是 subprocess 调 GPT-SoVITS 官方脚本，
所以上游升级时只要参数名不变就不用改本库。
"""
from .config import ConfigError, TrainerConfig
from .pipeline import PipelineError, run_pipeline
from .state import STAGES, STAGE_LABELS, Task, list_tasks

__version__ = "0.1.0"

__all__ = [
    "ConfigError",
    "PipelineError",
    "STAGES",
    "STAGE_LABELS",
    "Task",
    "TrainerConfig",
    "list_tasks",
    "run_pipeline",
]
