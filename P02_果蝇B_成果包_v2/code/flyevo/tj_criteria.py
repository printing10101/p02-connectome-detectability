"""TJ1 预注册判据装载器.

冻结 JSON (code/ 上一级的 TJ1-预注册判据-frozen-v1.0.json) 是唯一判据来源:
判决脚本启动时从这里取 experiments.<key>.criteria。文件或键缺失直接 raise ——
宁可不跑, 不许静默回退硬编码 (台账 F17 跳步的教训: 判据漂移必须当场响亮失败)。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # code/
CRITERIA_PATH = ROOT.parent / "TJ1-预注册判据-frozen-v1.0.json"

_CACHE: dict | None = None


class CriteriaError(RuntimeError):
    """判据文件/键不可用 —— 拒绝运行, 禁止回退硬编码。"""


def _raw() -> bytes:
    if not CRITERIA_PATH.exists():
        raise CriteriaError(
            f"预注册判据文件缺失: {CRITERIA_PATH} —— 判决实验拒绝运行 (不许回退硬编码)"
        )
    return CRITERIA_PATH.read_bytes()


def _doc() -> dict:
    global _CACHE
    if _CACHE is None:
        _CACHE = json.loads(_raw().decode("utf-8"))
    return _CACHE


def load(key: str, field: str = "criteria") -> dict:
    """取 experiments.<key>.<field>; 键/字段缺失时列出可用键, 响亮失败.

    field 默认 "criteria"; A2 阶梯的判据挂在 manifest_criteria 下, 用
    load(key, field="manifest_criteria") 取。
    """
    doc = _doc()
    exps = doc.get("experiments", {})
    if key not in exps:
        raise CriteriaError(
            f"判据键缺失: {key!r} 不在 {CRITERIA_PATH.name} 的 experiments 下 —— "
            f"可用键: {sorted(exps)} (拒绝运行, 不许回退硬编码)"
        )
    crit = exps[key].get(field)
    if crit is None:
        raise CriteriaError(
            f"判据键 {key!r} 下没有 {field!r} 字段 —— 拒绝运行 (不许回退硬编码)"
        )
    return dict(crit)


def sha256_of_json() -> str:
    """冻结判据文件的 sha256 (原字节), 供每份 verdict JSON 留痕."""
    return hashlib.sha256(_raw()).hexdigest()
