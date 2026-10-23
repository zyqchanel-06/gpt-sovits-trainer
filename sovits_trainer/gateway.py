# -*- coding: utf-8 -*-
"""把训练产物写进推理网关的配置。

这是「训练 → 使用」之间唯一的耦合点，格式就是 gpt-sovits-gateway 的 config.json：
给它加一条 voice，带上 refAudio / promptText / gpt / sovits 四个字段即可。
写之前会备份原文件，避免手滑覆盖用户的配置。
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class GatewayError(RuntimeError):
    pass


def load_config(path: Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise GatewayError("网关配置不存在：%s" % path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise GatewayError("网关配置不是合法 JSON（%s）：%s" % (path, exc)) from exc
    if not isinstance(data, dict):
        raise GatewayError("网关配置根节点必须是对象")
    return data


def backup(path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    target = Path(str(path) + ".bak-" + stamp)
    shutil.copy2(path, target)
    return target


def upsert_voice(
    config_path: Path,
    *,
    voice_id: str,
    sovits_path: str,
    gpt_path: str,
    ref_audio: str,
    prompt_text: str,
    prompt_lang: str = "zh",
    display_name: Optional[str] = None,
) -> Dict[str, Any]:
    """写入或更新一条声线；返回 {action, backup, voice}。"""
    config_path = Path(config_path)
    data = load_config(config_path)
    voices = data.get("voices")
    if not isinstance(voices, list):
        voices = []
    for item in voices:
        if not isinstance(item, dict):
            raise GatewayError("网关配置里 voices 含非对象项，拒绝修改")

    entry = {
        "id": voice_id,
        "name": display_name or voice_id,
        "lang": prompt_lang,
        "refAudio": ref_audio,
        "promptText": prompt_text,
        "promptLang": prompt_lang,
        "gpt": gpt_path,
        "sovits": sovits_path,
    }

    action = "added"
    for index, item in enumerate(voices):
        if item.get("id") == voice_id:
            voices[index] = {**item, **entry}
            action = "updated"
            break
    else:
        voices.append(entry)

    backup_path = backup(config_path)
    data["voices"] = voices
    if not data.get("defaultVoice"):
        data["defaultVoice"] = voice_id
    config_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"action": action, "backup": str(backup_path), "voice": entry}


def read_first_sample(list_path: Path) -> Optional[Dict[str, str]]:
    """从标注 .list 里取第一条，作为网关的参考音频/文本默认值。

    .list 的格式：<wav路径>|<说话人>|<语言>|<文本>
    """
    list_path = Path(list_path)
    if not list_path.exists():
        return None
    for line in list_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.count("|") < 3:
            continue
        parts = line.split("|")
        return {"wav": parts[0].strip(), "speaker": parts[1].strip(),
                "lang": parts[2].strip(), "text": parts[3].strip()}
    return None
