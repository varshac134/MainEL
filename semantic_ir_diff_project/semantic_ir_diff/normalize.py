from __future__ import annotations

import re
from typing import Dict


_METADATA_LINE = re.compile(r"^\s*!\d+\s*= .*", re.MULTILINE)
_METADATA_ATTACHMENT = re.compile(r",\s*!\w+\s*!\d+")
_DBG_ATTACHMENT = re.compile(r"!dbg\s*!\d+")


def _strip_metadata(text: str) -> str:
    text = re.sub(_METADATA_LINE, "", text)
    text = re.sub(_METADATA_ATTACHMENT, "", text)
    text = re.sub(_DBG_ATTACHMENT, "", text)
    return text


def _canonicalize_registers(text: str) -> str:
    reg_map: Dict[str, str] = {}
    counter = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal counter
        reg = match.group(0)
        if reg not in reg_map:
            reg_map[reg] = f"%r{counter}"
            counter += 1
        return reg_map[reg]

    return re.sub(r"(?<![0-9A-Za-z_])%\d+", replace, text)


def _canonicalize_labels(text: str) -> str:
    label_map: Dict[str, str] = {}
    counter = 0

    def replace_label(match: re.Match[str]) -> str:
        nonlocal counter
        label = match.group(1)
        if label not in label_map:
            label_map[label] = f"bb{counter}"
            counter += 1
        return f"{label_map[label]}:"

    text = re.sub(r"^\s*([A-Za-z$._][0-9A-Za-z$._]*)\s*:", replace_label, text, flags=re.MULTILINE)
    for original, canonical in label_map.items():
        text = re.sub(rf"\b{re.escape(original)}\b", canonical, text)
    return text


def _strip_comments(text: str) -> str:
    return re.sub(r";.*", "", text)


def normalize_ir_text(text: str) -> str:
    text = _strip_metadata(text)
    text = _strip_comments(text)
    text = _canonicalize_registers(text)
    text = _canonicalize_labels(text)
    lines = []
    for line in text.splitlines():
        stripped = re.sub(r"\s+", " ", line).strip()
        if stripped:
            lines.append(stripped)
    return "\n".join(lines)
