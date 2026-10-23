# -*- coding: utf-8 -*-
"""环境探测：GPU 显存、推理服务占用。

训练和推理抢同一块显存是这类流水线最常见的翻车点——
训练前提醒一句，比跑到一半 CUDA OOM 强。
"""
from __future__ import annotations

import shutil
import subprocess
from typing import Any, Dict, List, Optional


def nvidia_smi() -> Optional[str]:
    return shutil.which("nvidia-smi")


def gpu_info() -> List[Dict[str, Any]]:
    """返回每块卡的显存占用；没有 nvidia-smi 时返回空列表。"""
    if not nvidia_smi():
        return []
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if out.returncode != 0:
        return []
    cards = []
    for line in out.stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            cards.append({
                "index": int(parts[0]),
                "name": parts[1],
                "total_mb": int(float(parts[2])),
                "used_mb": int(float(parts[3])),
                "util_percent": int(float(parts[4])),
                "free_mb": int(float(parts[2]) - float(parts[3])),
            })
        except ValueError:
            continue
    return cards


def gateway_busy(gateway_url: str = "http://127.0.0.1:9881", timeout: float = 2.0) -> Optional[Dict[str, Any]]:
    """问一下推理网关最近有没有在忙；连不上返回 None（说明没开，也就不冲突）。"""
    import json
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(gateway_url.rstrip("/") + "/health", timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError):
        return None
    stats = data.get("stats") or {}
    return {
        "url": gateway_url,
        "upstream_up": data.get("upstreamUp"),
        "requests": stats.get("requests", 0),
        "last_ms": stats.get("lastMs", 0),
    }


def warn_if_contended(gateway_url: str = "http://127.0.0.1:9881") -> List[str]:
    """返回需要提醒用户的事项（空列表表示没风险）。"""
    warnings: List[str] = []
    cards = gpu_info()
    for card in cards:
        if card["used_mb"] > card["total_mb"] * 0.5:
            warnings.append(
                "GPU%d 已占用 %d/%d MB —— 训练可能 OOM，建议先停掉推理服务"
                % (card["index"], card["used_mb"], card["total_mb"])
            )
    busy = gateway_busy(gateway_url)
    if busy is not None:
        warnings.append(
            "推理网关在线（%s），训练会与它抢显存；建议先停掉网关再训练" % gateway_url
        )
    return warnings
