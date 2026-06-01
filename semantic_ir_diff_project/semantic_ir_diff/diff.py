from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, List, Tuple

from .parser import IRFunction


@dataclass
class FunctionDiff:
    name: str
    changed: bool
    added_blocks: int
    removed_blocks: int
    instruction_delta: int
    added_calls: List[str]
    removed_calls: List[str]
    branch_changed: bool
    facts: List[str]


@dataclass
class IRDiffResult:
    added_functions: List[str]
    removed_functions: List[str]
    changed_functions: List[FunctionDiff]


def _list_diff(old: List[str], new: List[str]) -> Tuple[List[str], List[str]]:
    old_set = set(old)
    new_set = set(new)
    return sorted(new_set - old_set), sorted(old_set - new_set)


def diff_functions(old_funcs: Dict[str, IRFunction], new_funcs: Dict[str, IRFunction]) -> IRDiffResult:
    added = sorted(name for name in new_funcs if name not in old_funcs)
    removed = sorted(name for name in old_funcs if name not in new_funcs)
    changed_list: List[FunctionDiff] = []

    for name in sorted(old_funcs.keys() & new_funcs.keys()):
        old = old_funcs[name]
        new = new_funcs[name]
        added_calls, removed_calls = _list_diff(sorted(old.calls), sorted(new.calls))
        added_blocks, removed_blocks = _list_diff(sorted(old.blocks), sorted(new.blocks))
        branch_changed = old.terminators != new.terminators
        instruction_delta = abs(new.instr_count - old.instr_count)
        changed = bool(added_calls or removed_calls or added_blocks or removed_blocks or branch_changed or instruction_delta)

        facts: List[str] = []
        if added_blocks:
            facts.append(f"added blocks: {', '.join(added_blocks)}")
        if removed_blocks:
            facts.append(f"removed blocks: {', '.join(removed_blocks)}")
        if added_calls:
            facts.append(f"new calls: {', '.join(added_calls)}")
        if removed_calls:
            facts.append(f"removed calls: {', '.join(removed_calls)}")
        if branch_changed:
            facts.append("terminator structure changed")
        if instruction_delta:
            sign = "increased" if new.instr_count > old.instr_count else "decreased"
            facts.append(f"instruction count {sign} by {instruction_delta}")

        if changed:
            changed_list.append(
                FunctionDiff(
                    name=name,
                    changed=changed,
                    added_blocks=len(added_blocks),
                    removed_blocks=len(removed_blocks),
                    instruction_delta=instruction_delta,
                    added_calls=added_calls,
                    removed_calls=removed_calls,
                    branch_changed=branch_changed,
                    facts=facts,
                )
            )

    return IRDiffResult(added_functions=added, removed_functions=removed, changed_functions=changed_list)
