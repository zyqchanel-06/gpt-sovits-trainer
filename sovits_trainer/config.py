# -*- coding: utf-8 -*-
"""训练器配置：GPT-SoVITS 仓库位置、解释器、任务目录。

设计原则：**不猜路径**。找不到就明确报错并给出怎么配，而不是默默用错目录。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

CONFIG_FILENAME = ".sovits-trainer.json"

# 找不到时依次探测的位置（都要求里面有 webui.py + GPT_SoVITS/）
COMMON_ROOTS = (
    "E:/DeepSeekHarness/tts/GPT-SoVITS-main",
    "D:/GPT-SoVITS",
    "E:/GPT-SoVITS",
    "D:/AI/GPT-SoVITS",
    str(Path.home() / "GPT-SoVITS"),
)


class ConfigError(RuntimeError):
    """配置不合法或找不到 GPT-SoVITS。"""


def looks_like_gpt_sovits(path: Path) -> bool:
    return (path / "webui.py").exists() and (path / "GPT_SoVITS").is_dir()


# 训练链路真正需要的第三方包（缺一个都跑不完整）
REQUIRED_MODULES = ("numpy", "torch", "scipy", "librosa", "yaml")
OPTIONAL_MODULES = ("pytorch_lightning", "funasr")


def detect_python() -> str:
    """默认用当前解释器：训练脚本和 trainer 共用同一套依赖最省事。"""
    return os.environ.get("SOVITS_TRAINER_PYTHON") or sys.executable


def probe_modules(python: str, modules=REQUIRED_MODULES) -> Dict[str, bool]:
    """问某个解释器：这些包装了没有。

    用子进程问，避免把训练依赖 import 进 trainer 自己的进程（可能全是缺失的）。
    """
    code = (
        "import importlib, json, sys\n"
        "out = {}\n"
        "for name in %r:\n"
        "    try:\n"
        "        importlib.import_module(name); out[name] = True\n"
        "    except Exception:\n"
        "        out[name] = False\n"
        "print(json.dumps(out))\n"
    ) % (tuple(modules),)
    try:
        result = subprocess.run(
            [str(python), "-c", code],
            capture_output=True, text=True, timeout=120,
        )
    except (OSError, subprocess.SubprocessError):
        return {name: False for name in modules}
    if result.returncode != 0:
        return {name: False for name in modules}
    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {name: False for name in modules}


@dataclass
class TrainerConfig:
    gpt_sovits_root: Path
    python: str = ""
    work_root: Path = field(default_factory=lambda: Path.cwd() / "voice-tasks")
    weights_dir: Optional[Path] = None
    version: str = "v2"
    device: str = "cuda"
    is_half: bool = True
    cuda_visible: str = "0"

    def __post_init__(self) -> None:
        self.gpt_sovits_root = Path(self.gpt_sovits_root).expanduser().resolve()
        self.work_root = Path(self.work_root).expanduser().resolve()
        if not self.python:
            self.python = detect_python()
        if self.weights_dir is None:
            self.weights_dir = self.gpt_sovits_root / "GPT_SoVITS" / "pretrained_models"
        else:
            self.weights_dir = Path(self.weights_dir).expanduser().resolve()

    # ---- 校验 ----
    def validate(self) -> None:
        if not looks_like_gpt_sovits(self.gpt_sovits_root):
            raise ConfigError(
                "这不像 GPT-SoVITS 仓库：%s\n"
                "需要同时存在 webui.py 和 GPT_SoVITS/ 目录。\n"
                "用 --gpt-sovits-root 指定，或设环境变量 GPT_SOVITS_ROOT。" % self.gpt_sovits_root
            )
        if not Path(self.python).exists() and self.python != sys.executable:
            raise ConfigError("Python 解释器不存在：%s" % self.python)

    # ---- 产物路径 ----
    def stage_dir(self, voice: str, stage: str) -> Path:
        """每个任务每步一个目录，互不干扰。"""
        return self.work_root / voice / stage

    def task_file(self, voice: str) -> Path:
        return self.work_root / voice / "task.json"

    def log_path(self, voice: str, stage: str) -> Path:
        return self.work_root / voice / "logs" / (stage + ".log")

    # ---- 序列化 ----
    def as_dict(self) -> Dict[str, Any]:
        return {
            "gpt_sovits_root": str(self.gpt_sovits_root),
            "python": self.python,
            "work_root": str(self.work_root),
            "weights_dir": str(self.weights_dir),
            "version": self.version,
            "device": self.device,
            "is_half": self.is_half,
            "cuda_visible": self.cuda_visible,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainerConfig":
        if "gpt_sovits_root" not in data:
            raise ConfigError("配置缺少 gpt_sovits_root")
        return cls(
            gpt_sovits_root=Path(data["gpt_sovits_root"]),
            python=str(data.get("python") or detect_python()),
            work_root=Path(data.get("work_root") or (Path.cwd() / "voice-tasks")),
            weights_dir=Path(data["weights_dir"]) if data.get("weights_dir") else None,
            version=str(data.get("version") or "v2"),
            device=str(data.get("device") or "cuda"),
            is_half=bool(data.get("is_half", True)),
            cuda_visible=str(data.get("cuda_visible") or "0"),
        )

    @classmethod
    def load(cls, path: Optional[str] = None) -> "TrainerConfig":
        """优先级：显式路径 > 环境变量 GPT_SOVITS_ROOT > 当前目录配置 > 常见位置。"""
        candidates = []
        if path:
            candidates.append(Path(path))
        env_root = os.environ.get("GPT_SOVITS_ROOT")
        if env_root:
            root_path = Path(env_root)
            if looks_like_gpt_sovits(root_path):
                return cls(gpt_sovits_root=root_path)
        candidates.append(Path.cwd() / CONFIG_FILENAME)
        for candidate in candidates:
            if candidate.exists():
                data = json.loads(candidate.read_text(encoding="utf-8"))
                return cls.from_dict(data)
        for candidate in COMMON_ROOTS:
            root_path = Path(candidate)
            if looks_like_gpt_sovits(root_path):
                return cls(gpt_sovits_root=root_path)
        raise ConfigError(
            "找不到 GPT-SoVITS 仓库。请用 --gpt-sovits-root 指定，"
            "或设置环境变量 GPT_SOVITS_ROOT，或在当前目录放一个 %s。" % CONFIG_FILENAME
        )

    def save(self, path: Optional[str] = None) -> Path:
        target = Path(path) if path else (Path.cwd() / CONFIG_FILENAME)
        target.write_text(json.dumps(self.as_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return target
