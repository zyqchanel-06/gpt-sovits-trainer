# -*- coding: utf-8 -*-
"""流水线编排：按固定顺序跑六个阶段，支持断点续跑和单步执行。"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from . import stages
from .config import TrainerConfig
from .runner import StepResult
from .state import STAGES, STAGE_LABELS, Task


class PipelineError(RuntimeError):
    pass


def run_stage(
    stage: str,
    config: TrainerConfig,
    task: Task,
    options: Dict[str, Any],
) -> Optional[str]:
    """跑单个阶段，返回 artifact 路径（没产物就返回 None）。失败抛 PipelineError。"""

    if stage == "slice":
        audio = options.get("audio_path")
        if not audio:
            raise PipelineError("缺少音频输入：init 时用 --audio 指定，或运行加 --audio")
        result = stages.slice.run(
            config, task, Path(audio),
            threshold=options.get("threshold", -34),
            min_length=options.get("min_length", 4000),
            min_interval=options.get("min_interval", 300),
            max_sil_kept=options.get("max_sil_kept", 500),
            max_length=options.get("max_length", 10000),
            alpha=options.get("alpha", 0.25),
        )
        if not result.ok:
            raise PipelineError("切片失败：" + " / ".join(result.errors or [])[-300:])
        info = stages.slice.collect_output(config, task)
        if info["count"] == 0:
            raise PipelineError("切片后没有产生任何 wav，检查输入音频和阈值")
        return info["dir"]

    if stage == "asr":
        outcome = stages.asr.run(
            config, task,
            provided_list=options.get("list_path"),
            backend=options.get("asr_backend", "faster-whisper"),
            model_size=options.get("asr_model", "large-v3"),
            language=options.get("language", "zh"),
            precision=options.get("asr_precision", "float16"),
        )
        if outcome.get("skipped"):
            return outcome["list_path"]
        result = outcome.get("result")
        if result is not None and not result.ok:
            raise PipelineError("ASR 失败：" + " / ".join(result.errors or [])[-300:])
        if not outcome.get("list_path"):
            raise PipelineError("ASR 没有产出 .list 文件，检查日志：" + str(config.log_path(task.voice, "asr")))
        return outcome["list_path"]

    if stage == "prep":
        # 标注来源优先级：显式参数 > 上一步（asr）登记的产物 > init 时存的默认值
        list_path = (
            options.get("resolved_list")
            or task.stages["asr"].artifact
            or options.get("list_path")
        )
        if not list_path:
            raise PipelineError("缺少标注文件：先跑 asr 阶段，或用 --list 指定")
        sliced = str(config.stage_dir(task.voice, "slice") / "wav")
        results = stages.prep.run(config, task, Path(list_path), Path(sliced))
        bad = [r for r in results if not r.ok]
        if bad:
            raise PipelineError("特征提取失败：" + " / ".join(bad[0].errors or [])[-300:])
        found = stages.prep.artifacts(config, task)
        missing = [key for key, value in found.items() if value is None]
        if missing:
            raise PipelineError("特征提取后缺少产物：%s" % ", ".join(missing))
        return found["semantic"]

    if stage == "s2":
        result = stages.s2.run(
            config, task,
            batch_size=options.get("s2_batch_size", 4),
            epochs=options.get("s2_epochs", 8),
            save_every_epoch=options.get("save_every_epoch", 4),
        )
        if not result.ok:
            raise PipelineError("SoVITS 训练失败：" + " / ".join(result.errors or [])[-300:])
        return None

    if stage == "s1":
        result = stages.s1.run(
            config, task,
            batch_size=options.get("s1_batch_size", 4),
            epochs=options.get("s1_epochs", 8),
            save_every_epoch=options.get("save_every_epoch", 4),
        )
        if not result.ok:
            raise PipelineError("GPT 训练失败：" + " / ".join(result.errors or [])[-300:])
        return None

    if stage == "register":
        outcome = stages.register.run(config, task, epoch=options.get("register_epoch"))
        options["register_outcome"] = outcome
        return outcome["voices_file"]

    raise PipelineError("未知阶段：%s" % stage)


def run_pipeline(
    config: TrainerConfig,
    task: Task,
    options: Dict[str, Any],
    *,
    only: Optional[str] = None,
    from_stage: Optional[str] = None,
    force: bool = False,
    echo: bool = False,
) -> Dict[str, Any]:
    """按顺序执行；已完成的阶段自动跳过（force=True 时重跑）。"""
    if only and only not in STAGES:
        raise PipelineError("未知阶段：%s（可选：%s）" % (only, ", ".join(STAGES)))
    if from_stage and from_stage not in STAGES:
        raise PipelineError("未知阶段：%s" % from_stage)

    started = from_stage is not None
    executed = []

    for stage in STAGES:
        if only and stage != only:
            continue
        if from_stage and not started:
            if stage == from_stage:
                started = True
            else:
                continue
        if task.is_done(stage) and not (force or only or from_stage):
            continue

        task.start(stage)
        try:
            artifact = run_stage(stage, config, task, options)
        except Exception as exc:  # noqa: BLE001 - 统一转成状态记录
            task.fail(stage, str(exc))
            raise PipelineError("[%s %s] %s" % (stage, STAGE_LABELS[stage], exc)) from exc
        task.finish(stage, artifact)
        executed.append(stage)

    return {"executed": executed, "task": task, "register": options.get("register_outcome")}
