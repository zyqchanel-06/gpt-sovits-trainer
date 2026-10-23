# -*- coding: utf-8 -*-
"""任务状态机：六步流水线 + JSON 持久化，支持断点续跑。

为什么要持久化：训练动辄几十分钟，中途挂了（显存爆、断电、Ctrl+C）不该前功尽弃。
每步开始/结束都写盘，重跑时自动跳过已完成的步骤。
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# 流水线顺序固定：后面的步骤依赖前面的产物
STAGES = ("slice", "asr", "prep", "s2", "s1", "register")

STAGE_LABELS = {
    "slice": "音频切片",
    "asr": "语音识别标注",
    "prep": "特征提取",
    "s2": "训练 SoVITS",
    "s1": "训练 GPT",
    "register": "登记产物",
}

STATUS_ICONS = {
    "pending": "·",
    "running": "…",
    "done": "✓",
    "failed": "✗",
    "skipped": "-",
}


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


@dataclass
class StageRecord:
    status: str = "pending"
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error: Optional[str] = None
    progress: Optional[str] = None
    artifact: Optional[str] = None

    @property
    def terminal(self) -> bool:
        return self.status in ("done", "skipped", "failed")


@dataclass
class Task:
    voice: str
    root: Path
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    stages: Dict[str, StageRecord] = field(default_factory=lambda: {name: StageRecord() for name in STAGES})
    meta: Dict[str, Any] = field(default_factory=dict)

    # ---- 读写 ----
    @classmethod
    def path_for(cls, root: Path, voice: str) -> Path:
        return Path(root) / voice / "task.json"

    @classmethod
    def load(cls, root: Path, voice: str) -> Optional["Task"]:
        path = cls.path_for(root, voice)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        stages = {}
        for name in STAGES:
            raw = (data.get("stages") or {}).get(name) or {}
            stages[name] = StageRecord(
                status=raw.get("status", "pending"),
                started_at=raw.get("started_at"),
                finished_at=raw.get("finished_at"),
                error=raw.get("error"),
                progress=raw.get("progress"),
                artifact=raw.get("artifact"),
            )
        return cls(
            voice=data["voice"],
            root=Path(root),
            created_at=data.get("created_at", _now()),
            updated_at=data.get("updated_at", _now()),
            stages=stages,
            meta=data.get("meta") or {},
        )

    def save(self) -> Path:
        self.updated_at = _now()
        path = self.path_for(self.root, self.voice)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "voice": self.voice,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "stages": {name: asdict(record) for name, record in self.stages.items()},
            "meta": self.meta,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return path

    # ---- 状态迁移 ----
    def start(self, stage: str) -> StageRecord:
        record = self.stages[stage]
        record.status = "running"
        record.started_at = _now()
        record.finished_at = None
        record.error = None
        self.save()
        return record

    def finish(self, stage: str, artifact: Optional[str] = None) -> StageRecord:
        record = self.stages[stage]
        record.status = "done"
        record.finished_at = _now()
        record.error = None
        if artifact:
            record.artifact = artifact
        self.save()
        return record

    def fail(self, stage: str, error: str) -> StageRecord:
        record = self.stages[stage]
        record.status = "failed"
        record.finished_at = _now()
        record.error = error
        self.save()
        return record

    def skip(self, stage: str, reason: str = "") -> StageRecord:
        record = self.stages[stage]
        record.status = "skipped"
        record.finished_at = _now()
        record.progress = reason or None
        self.save()
        return record

    def set_progress(self, stage: str, text: str) -> None:
        self.stages[stage].progress = text
        self.save()

    # ---- 查询 ----
    def is_done(self, stage: str) -> bool:
        return self.stages[stage].status in ("done", "skipped")

    def next_pending(self) -> Optional[str]:
        for name in STAGES:
            if self.stages[name].status not in ("done", "skipped"):
                return name
        return None

    def reset_from(self, stage: str, purge_dir=None) -> List[str]:
        """把 stage 及其之后的所有步骤重置为 pending（重跑用）。

        purge_dir(stage) -> Path 被传入时会**删除该阶段的产物目录**。
        这一步很关键：GPT-SoVITS 的预处理脚本是增量式的（产物存在就整段跳过），
        只重置状态而不删产物，会出现「看起来重跑了、其实用的还是旧数据」。
        """
        import shutil

        reset = []
        hit = False
        for name in STAGES:
            if name == stage:
                hit = True
            if hit:
                self.stages[name] = StageRecord()
                reset.append(name)
                if purge_dir is not None:
                    try:
                        target = Path(purge_dir(name))
                        if target.exists():
                            shutil.rmtree(target)
                    except OSError:
                        pass
        self.save()
        return reset

    def summary(self) -> str:
        lines = ["任务: %s" % self.voice]
        for name in STAGES:
            record = self.stages[name]
            icon = STATUS_ICONS.get(record.status, "?")
            extra = []
            if record.progress:
                extra.append(record.progress)
            if record.error:
                extra.append("错误: " + record.error.splitlines()[-1][:100])
            if record.artifact:
                extra.append(Path(record.artifact).name)
            lines.append("  %s %-10s %-14s %s" % (
                icon, name, STAGE_LABELS[name], " · ".join(extra)
            ))
        return "\n".join(lines)


def list_tasks(root: Path) -> List[Task]:
    root = Path(root)
    if not root.exists():
        return []
    tasks = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        task = Task.load(root, child.name)
        if task is not None:
            tasks.append(task)
    return tasks
