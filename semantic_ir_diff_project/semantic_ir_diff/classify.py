from __future__ import annotations

from typing import List

from .diff import FunctionDiff
from .parser import IRFunction


def _vectorization_change(old: IRFunction, new: IRFunction) -> List[str]:
    results: List[str] = []
    if old.has_loop_vectorize and not new.has_loop_vectorize:
        results.append("lost loop vectorization hints")
    elif not old.has_loop_vectorize and new.has_loop_vectorize:
        results.append("gained loop vectorization hints")
    return results


def _unroll_change(old: IRFunction, new: IRFunction) -> List[str]:
    results: List[str] = []
    if old.has_loop_unroll and not new.has_loop_unroll:
        results.append("lost loop unroll metadata")
    elif not old.has_loop_unroll and new.has_loop_unroll:
        results.append("gained loop unroll metadata")
    return results


def _inlining_change(old: IRFunction, new: IRFunction) -> List[str]:
    results: List[str] = []
    if old.calls and not new.calls:
        results.append("calls removed; function may have been inlined")
    elif not old.calls and new.calls:
        results.append("calls added; function may have been extracted or de-inlined")
    return results


def _memory_change(old: IRFunction, new: IRFunction) -> List[str]:
    results: List[str] = []
    if old.load_count != new.load_count:
        results.append(
            f"load count {'increased' if new.load_count > old.load_count else 'decreased'} by {abs(new.load_count - old.load_count)}"
        )
    if old.store_count != new.store_count:
        results.append(
            f"store count {'increased' if new.store_count > old.store_count else 'decreased'} by {abs(new.store_count - old.store_count)}"
        )
    return results


def classify_function_diff(old: IRFunction, new: IRFunction, function_diff: FunctionDiff) -> List[str]:
    categories: List[str] = []
    categories.extend(_vectorization_change(old, new))
    categories.extend(_unroll_change(old, new))
    categories.extend(_inlining_change(old, new))
    categories.extend(_memory_change(old, new))
    if function_diff.branch_changed:
        categories.append("control-flow changed / branch shape changed")
    if function_diff.added_blocks or function_diff.removed_blocks:
        categories.append("basic block structure changed")
    if function_diff.instruction_delta:
        categories.append("instruction count changed")
    return categories
