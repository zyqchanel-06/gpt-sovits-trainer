# -*- coding: utf-8 -*-
"""阶段 6：登记产物。

扫描权重目录，找出这次训练产出的 .pth / .ckpt，写进 voices.json。
这个文件是训练器与推理网关之间的交接单，gateway sync 就吃它。
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import TrainerConfig
from ..state import Task

# SoVITS: <voice>_e8_s248.pth    GPT: <voice>-e8.ckpt
PTH_PATTERN = re.compile(r"^(?P<name>.+)_e(?P<epoch>\d+)_s(?P<step>\d+)\.pth$", re.IGNORECASE)
CKPT_PATTERN = re.compile(r"^(?P<name>.+)-e(?P<epoch>\d+)\.ckpt$", re.IGNORECASE)


def scan(weights_dir: Path, voice: str) -> Dict[str, List[Dict[str, Any]]]:
    """在权重目录里找某个声线的所有训练产物。"""
    weights_dir = Path(weights_dir)
    sovits: List[Dict[str, Any]] = []
    gpt: List[Dict[str, Any]] = []
    if not weights_dir.exists():
        return {"sovits": sovits, "gpt": gpt}

    for path in weights_dir.iterdir():
        if not path.is_file():
            continue
        pth = PTH_PATTERN.match(path.name)
        if pth and pth.group("name") == voice:
            sovits.append({
                "path": str(path),
                "name": path.name,
                "epoch": int(pth.group("epoch")),
                "step": int(pth.group("step")),
                "size_mb": round(path.stat().st_size / 1024 / 1024, 1),
            })
            continue
        ckpt = CKPT_PATTERN.match(path.name)
        if ckpt and ckpt.group("name") == voice:
            gpt.append({
                "path": str(path),
                "name": path.name,
                "epoch": int(ckpt.group("epoch")),
                "size_mb": round(path.stat().st_size / 1024 / 1024, 1),
            })

    sovits.sort(key=lambda item: (item["epoch"], item["step"]))
    gpt.sort(key=lambda item: item["epoch"])
    return {"sovits": sovits, "gpt": gpt}


def pick_best(found: Dict[str, List[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    """默认取 epoch 最大的那组（通常收敛最好；想换档位就显式指定 --epoch）。"""
    sovits = found.get("sovits") or []
    gpt = found.get("gpt") or []
    if not sovits or not gpt:
        return None
    best_sovits = sovits[-1]
    best_gpt = gpt[-1]
    return {
        "sovits": best_sovits["path"],
        "gpt": best_gpt["path"],
        "epoch": min(best_sovits["epoch"], best_gpt["epoch"]),
        "sovits_name": best_sovits["name"],
        "gpt_name": best_gpt["name"],
    }


def write_voices(config: TrainerConfig, task: Task, chosen: Dict[str, Any]) -> Path:
    """写出交接单。多次训练同一 voice 会覆盖最新一次。"""
    target = config.work_root / task.voice / "voices.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "voice": task.voice,
        "updated_at": task.updated_at,
        "sovits": chosen["sovits"],
        "gpt": chosen["gpt"],
        "epoch": chosen["epoch"],
    }
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target


def run(config: TrainerConfig, task: Task, *, epoch: Optional[int] = None) -> Dict[str, Any]:
    found = scan(config.weights_dir, task.voice)
    if not found["sovits"] or not found["gpt"]:
        raise RuntimeError(
            "在 %s 里没找到 %s 的训练产物（需要同时有 %s_e<N>_s<M>.pth 和 %s-e<N>.ckpt）"
            % (config.weights_dir, task.voice, task.voice, task.voice)
        )

    if epoch is not None:
        sovits = [item for item in found["sovits"] if item["epoch"] == epoch]
        gpt = [item for item in found["gpt"] if item["epoch"] == epoch]
        if not sovits or not gpt:
            available = sorted({item["epoch"] for item in found["sovits"]})
            raise RuntimeError("epoch %s 没有产物；可用 epoch：%s" % (epoch, available))
        chosen = {
            "sovits": sovits[-1]["path"],
            "gpt": gpt[-1]["path"],
            "epoch": epoch,
            "sovits_name": sovits[-1]["name"],
            "gpt_name": gpt[-1]["name"],
        }
    else:
        chosen = pick_best(found)

    path = write_voices(config, task, chosen)
    return {"chosen": chosen, "voices_file": str(path), "all": found}
