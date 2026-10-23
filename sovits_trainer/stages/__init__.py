# -*- coding: utf-8 -*-
"""六个流水线阶段。

每个阶段都是 (config, task, **options) -> artifact 路径或 None 的纯函数，
方便单独测试和断点续跑。全部通过 subprocess 调用 GPT-SoVITS 官方脚本，
不在本库里重写任何训练逻辑。
"""
from . import asr, prep, register, s1, s2, slice  # noqa: F401

__all__ = ["slice", "asr", "prep", "s2", "s1", "register"]
