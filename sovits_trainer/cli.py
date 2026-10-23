# -*- coding: utf-8 -*-
"""命令行入口：doctor / init / run / status / list / reset / export / sync。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import gateway as gateway_mod
from . import gpu
from .config import ConfigError, TrainerConfig
from .pipeline import PipelineError, run_pipeline
from .state import STAGES, STAGE_LABELS, STATUS_ICONS, Task, list_tasks

DEFAULTS_KEY = "defaults"


# ---------------------------------------------------------------- 辅助
def _config(args: argparse.Namespace) -> TrainerConfig:
    config = TrainerConfig.load(getattr(args, "config", None))
    if getattr(args, "gpt_sovits_root", None):
        config = TrainerConfig(gpt_sovits_root=Path(args.gpt_sovits_root))
    if getattr(args, "work_root", None):
        config.work_root = Path(args.work_root).expanduser().resolve()
    if getattr(args, "weights_dir", None):
        config.weights_dir = Path(args.weights_dir).expanduser().resolve()
    config.validate()
    return config


def _require_task(config: TrainerConfig, voice: str) -> Task:
    task = Task.load(config.work_root, voice)
    if task is None:
        raise SystemExit(
            "任务不存在：%s（先跑 sovits-train init --voice %s --audio <音频>）" % (voice, voice)
        )
    return task


def _options(args: argparse.Namespace, task: Task) -> Dict[str, Any]:
    """命令行的参数 + init 时存下的默认值。"""
    meta = dict(task.meta or {})
    defaults = dict(meta.get(DEFAULTS_KEY) or {})
    options: Dict[str, Any] = dict(defaults)
    for key in (
        "audio_path", "list_path", "language", "asr_backend", "asr_model", "asr_precision",
        "threshold", "min_length", "min_interval", "max_sil_kept", "max_length", "alpha",
        "s2_batch_size", "s2_epochs", "s1_batch_size", "s1_epochs", "save_every_epoch",
        "register_epoch",
    ):
        value = getattr(args, key, None)
        if value is not None:
            options[key] = value
    return options


# ---------------------------------------------------------------- 子命令
def cmd_doctor(args: argparse.Namespace) -> int:
    print("== 环境检查 ==")
    try:
        config = _config(args)
    except ConfigError as exc:
        print("✗ 配置：", exc)
        return 2

    print("✓ GPT-SoVITS：", config.gpt_sovits_root)
    print("✓ Python    ：", config.python)
    print("✓ 权重目录  ：", config.weights_dir)
    print("✓ 任务目录  ：", config.work_root)

    for rel in ("webui.py", "GPT_SoVITS/s2_train.py", "GPT_SoVITS/s1_train.py",
                "tools/slice_audio.py", "tools/asr/fasterwhisper_asr.py",
                "GPT_SoVITS/configs/s2.json"):
        path = config.gpt_sovits_root / rel
        print(("  %s %s" % ("✓" if path.exists() else "✗", rel)))

    cards = gpu.gpu_info()
    if cards:
        for card in cards:
            print("  GPU%d %s：%d/%d MB 已用（空闲 %d MB）"
                  % (card["index"], card["name"], card["used_mb"], card["total_mb"], card["free_mb"]))
    else:
        print("  · 没检测到 nvidia-smi（CPU 训练会非常慢）")

    for warning in gpu.warn_if_contended():
        print("  ! " + warning)
    return 0


def cmd_init(args: argparse.Namespace) -> int:
    config = _config(args)
    voice = args.voice.strip()
    if not voice or not voice.replace("-", "").replace("_", "").isalnum():
        print("音色名只能包含字母、数字、下划线和连字符")
        return 2

    existing = Task.load(config.work_root, voice)
    task = existing or Task(voice=voice, root=config.work_root)
    defaults = dict(task.meta.get(DEFAULTS_KEY) or {})
    if args.audio:
        audio = Path(args.audio).expanduser().resolve()
        if not audio.exists():
            print("✗ 音频不存在：", audio)
            return 2
        defaults["audio_path"] = str(audio)
    if args.list:
        list_path = Path(args.list).expanduser().resolve()
        if not list_path.exists():
            print("✗ 标注文件不存在：", list_path)
            return 2
        defaults["list_path"] = str(list_path)
    for key in ("language", "asr_backend", "asr_model"):
        value = getattr(args, key, None)
        if value:
            defaults[key] = value
    task.meta[DEFAULTS_KEY] = defaults
    path = task.save()
    print(("已更新" if existing else "已创建") + " 任务：%s" % voice)
    print("  状态文件：%s" % path)
    if not defaults.get("audio_path") and not defaults.get("list_path"):
        print("  提示：还没指定音频。用 --audio <录音或目录> 或 --list <已有标注>")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    config = _config(args)
    task = _require_task(config, args.voice)
    options = _options(args, task)

    warnings = gpu.warn_if_contended()
    for warning in warnings:
        print("! " + warning)

    if not options.get("audio_path") and not options.get("list_path"):
        print("✗ 没有音频也没有标注，无法开始。先用 init --audio 指定")
        return 2

    try:
        outcome = run_pipeline(
            config, task, options,
            only=args.only, from_stage=args.from_stage, force=args.force, echo=args.verbose,
        )
    except PipelineError as exc:
        print("✗ " + str(exc))
        print(task.summary())
        return 1

    print("完成。执行的阶段：", ", ".join(outcome["executed"]) or "（全部已完成，跳过）")
    print(task.summary())

    register = outcome.get("register")
    if register:
        chosen = register["chosen"]
        print("\n产物：")
        print("  SoVITS: %s" % chosen["sovits_name"])
        print("  GPT   : %s" % chosen["gpt_name"])
        print("  交接单: %s" % register["voices_file"])
        if args.gateway:
            _do_sync(args, config, task, chosen, register["voices_file"])
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    config = _config(args)
    if args.all or not args.voice:
        tasks = list_tasks(config.work_root)
        if not tasks:
            print("还没有任何任务。用 sovits-train init 创建一个。")
            return 0
        for task in tasks:
            pending = task.next_pending()
            print("%s  %s" % (task.voice, "全部完成" if pending is None else "下一步：" + pending))
        return 0
    task = _require_task(config, args.voice)
    print(task.summary())
    if args.json:
        print(json.dumps({name: record.status for name, record in task.stages.items()},
                         ensure_ascii=False, indent=2))
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    return cmd_status(argparse.Namespace(**{**vars(args), "all": True, "voice": None}))


def cmd_reset(args: argparse.Namespace) -> int:
    config = _config(args)
    task = _require_task(config, args.voice)
    reset = task.reset_from(args.stage)
    print("已重置：", ", ".join(reset))
    print(task.summary())
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    config = _config(args)
    task = _require_task(config, args.voice)
    from .stages import register as register_stage

    try:
        outcome = register_stage.run(config, task, epoch=args.epoch)
    except Exception as exc:  # noqa: BLE001
        print("✗ " + str(exc))
        return 1
    chosen = outcome["chosen"]
    print("选出：")
    print("  SoVITS: %s (epoch %s)" % (chosen["sovits_name"], chosen["epoch"]))
    print("  GPT   : %s" % chosen["gpt_name"])
    print("  交接单: %s" % outcome["voices_file"])
    print("\n可用 epoch：%s" % sorted({item["epoch"] for item in outcome["all"]["sovits"]}))
    return 0


def _do_sync(
    args: argparse.Namespace,
    config: TrainerConfig,
    task: Task,
    chosen: Dict[str, Any],
    voices_file: str,
) -> None:
    gateway_path = Path(args.gateway).expanduser().resolve()
    list_path = Path(chosen.get("list_path") or "")
    sample = None
    candidates = [task.meta.get(DEFAULTS_KEY, {}).get("list_path"), voices_file]
    for candidate in candidates:
        if candidate and Path(candidate).suffix == ".list" and Path(candidate).exists():
            sample = gateway_mod.read_first_sample(Path(candidate))
            break
    if sample is None:
        # 从 prep 目录里找找 ASR/用户提供的标注
        for found in (config.work_root / task.voice).rglob("*.list"):
            sample = gateway_mod.read_first_sample(found)
            if sample:
                break
    if sample is None:
        print("! 找不到标注文件，无法自动填参考音频/文本")
        print("  请用 --ref-audio 和 --ref-text 手动指定（网关的 voice 必须有这两个字段）")
        return

    try:
        result = gateway_mod.upsert_voice(
            gateway_path,
            voice_id=task.voice,
            sovits_path=chosen["sovits"],
            gpt_path=chosen["gpt"],
            ref_audio=sample["wav"],
            prompt_text=sample["text"],
            prompt_lang=sample["lang"] or "zh",
        )
    except gateway_mod.GatewayError as exc:
        print("✗ 写入网关失败：", exc)
        return
    print("网关配置已%s：%s（备份 %s）"
          % ("更新" if result["action"] == "updated" else "新增", gateway_path, result["backup"]))


def cmd_sync(args: argparse.Namespace) -> int:
    config = _config(args)
    task = _require_task(config, args.voice)
    from .stages import register as register_stage

    try:
        outcome = register_stage.run(config, task, epoch=args.epoch)
    except Exception as exc:  # noqa: BLE001
        print("✗ " + str(exc))
        return 1
    _do_sync(args, config, task, outcome["chosen"], outcome["voices_file"])
    return 0


# ---------------------------------------------------------------- 解析器
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sovits-train",
        description="GPT-SoVITS 训练编排器：录音 → 六步流水线 → 可直接用的音色权重",
    )
    parser.add_argument("--config", help="训练器配置文件（默认 ./.sovits-trainer.json）")
    parser.add_argument("--gpt-sovits-root", help="GPT-SoVITS 仓库根目录")
    parser.add_argument("--work-root", help="任务目录（默认 ./voice-tasks）")
    parser.add_argument("--weights-dir", help="权重输出目录")
    sub = parser.add_subparsers(dest="command", required=True)

    p_doctor = sub.add_parser("doctor", help="检查环境是否就绪")
    p_doctor.set_defaults(func=cmd_doctor)

    p_init = sub.add_parser("init", help="创建或更新任务")
    p_init.add_argument("--voice", required=True, help="音色名（字母数字下划线连字符）")
    p_init.add_argument("--audio", help="原始录音（文件或目录）")
    p_init.add_argument("--list", help="已有标注 .list（跳过 ASR）")
    p_init.add_argument("--language", default=None, help="语言，默认 zh")
    p_init.add_argument("--asr-backend", dest="asr_backend", choices=["faster-whisper", "funasr"])
    p_init.add_argument("--asr-model", dest="asr_model", help="ASR 模型，默认 large-v3")
    p_init.set_defaults(func=cmd_init)

    p_run = sub.add_parser("run", help="跑流水线")
    p_run.add_argument("--voice", required=True)
    p_run.add_argument("--audio", dest="audio_path", help="覆盖 init 时的音频")
    p_run.add_argument("--list", dest="list_path", help="覆盖标注文件")
    p_run.add_argument("--only", choices=list(STAGES), help="只跑某一个阶段")
    p_run.add_argument("--from", dest="from_stage", choices=list(STAGES), help="从某阶段开始重跑")
    p_run.add_argument("--force", action="store_true", help="忽略已完成状态，全部重跑")
    p_run.add_argument("--verbose", action="store_true", help="把训练输出同时打到终端")
    p_run.add_argument("--s2-epochs", dest="s2_epochs", type=int)
    p_run.add_argument("--s1-epochs", dest="s1_epochs", type=int)
    p_run.add_argument("--s2-batch-size", dest="s2_batch_size", type=int)
    p_run.add_argument("--s1-batch-size", dest="s1_batch_size", type=int)
    p_run.add_argument("--save-every-epoch", dest="save_every_epoch", type=int)
    p_run.add_argument("--gateway", help="训练完成后自动写进这个网关配置")
    p_run.set_defaults(func=cmd_run)

    p_status = sub.add_parser("status", help="查看任务进度")
    p_status.add_argument("--voice")
    p_status.add_argument("--all", action="store_true")
    p_status.add_argument("--json", action="store_true")
    p_status.set_defaults(func=cmd_status)

    p_list = sub.add_parser("list", help="列出所有任务")
    p_list.set_defaults(func=cmd_list)

    p_reset = sub.add_parser("reset", help="重置某阶段及之后")
    p_reset.add_argument("--voice", required=True)
    p_reset.add_argument("--stage", required=True, choices=list(STAGES))
    p_reset.set_defaults(func=cmd_reset)

    p_export = sub.add_parser("export", help="重新登记产物（可指定 epoch）")
    p_export.add_argument("--voice", required=True)
    p_export.add_argument("--epoch", type=int)
    p_export.set_defaults(func=cmd_export)

    p_sync = sub.add_parser("sync", help="把产物写进推理网关配置")
    p_sync.add_argument("--voice", required=True)
    p_sync.add_argument("--gateway", required=True, help="网关 config.json 路径")
    p_sync.add_argument("--epoch", type=int)
    p_sync.set_defaults(func=cmd_sync)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except ConfigError as exc:
        print("配置错误：", exc, file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\n已中断", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
