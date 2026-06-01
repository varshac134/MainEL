from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


CALL_RE = re.compile(r"call\s+[^@]*@([A-Za-z_.$][0-9A-Za-z_.$]*)\(")
FUNC_RE = re.compile(r"^define .* @([A-Za-z_.$][0-9A-Za-z_.$]*)\(")
LOOP_UNROLL_RE = re.compile(r"llvm\.loop\.unroll\.")
LOOP_VECTORIZE_RE = re.compile(r"llvm\.loop\.vectorize\.")


@dataclass
class IRFunction:
    name: str
    blocks: Dict[str, List[str]] = field(default_factory=dict)
    calls: Set[str] = field(default_factory=set)
    terminators: Dict[str, List[str]] = field(default_factory=dict)
    instr_count: int = 0
    load_count: int = 0
    store_count: int = 0
    has_loop_unroll: bool = False
    has_loop_vectorize: bool = False

    def __post_init__(self) -> None:
        self.load_count = 0
        self.store_count = 0


def parse_functions(ir_text: str) -> Dict[str, IRFunction]:
    functions: Dict[str, IRFunction] = {}
    current: Optional[IRFunction] = None
    current_block: Optional[str] = None
    lines = ir_text.splitlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        func_match = FUNC_RE.match(line)
        if func_match:
            current = IRFunction(name=func_match.group(1))
            functions[current.name] = current
            current_block = None
            continue

        if current is None:
            continue

        if line.endswith(":"):
            current_block = line[:-1]
            current.blocks[current_block] = []
            continue

        if current_block is None:
            continue

        current.blocks[current_block].append(line)
        current.instr_count += 1
        current.calls.update(CALL_RE.findall(line))
        if line.startswith("load "):
            current.load_count += 1
        elif line.startswith("store "):
            current.store_count += 1
        if line.startswith("br ") or line.startswith("switch ") or line.startswith("ret "):
            current.terminators[current_block] = [line]
        if LOOP_UNROLL_RE.search(line):
            current.has_loop_unroll = True
        if LOOP_VECTORIZE_RE.search(line):
            current.has_loop_vectorize = True

    return functions
