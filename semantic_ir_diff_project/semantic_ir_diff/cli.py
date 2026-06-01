from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .compiler import compile_to_ir
from .diff import diff_functions
from .classify import classify_function_diff
from .normalize import normalize_ir_text
from .parser import parse_functions


console = Console()


@click.command()
@click.argument("old_path", type=click.Path(exists=True))
@click.argument("new_path", type=click.Path(exists=True))
@click.option("--old-ir", is_flag=True, default=False, help="Treat the old path as LLVM IR and skip compilation.")
@click.option("--new-ir", is_flag=True, default=False, help="Treat the new path as LLVM IR and skip compilation.")
@click.option("--clang-path", type=click.Path(exists=True), default=None, help="Explicit clang/clang++ executable path.")
@click.option("--opt-level", default="-O3", help="Optimization level for LLVM IR emission.")
@click.option("--extra-compile-arg", multiple=True, help="Extra compile arguments to pass to clang.")
@click.option("--report-detail", type=click.Choice(["summary", "functions", "all"]), default="functions", help="Choose verbosity of the semantic diff report.")
def main(
    old_path: str,
    new_path: str,
    old_ir: bool,
    new_ir: bool,
    clang_path: Optional[str],
    opt_level: str,
    extra_compile_arg: tuple[str, ...],
    report_detail: str,
) -> None:
    """Compare two source or IR revisions and report semantic LLVM IR differences."""
    with tempfile.TemporaryDirectory() as temp_dir:
        old_ir_path = old_path
        new_ir_path = new_path

        if not old_ir:
            old_ir_path = Path(temp_dir) / "old.ll"
            compile_to_ir(old_path, str(old_ir_path), clang_path=clang_path, opt_level=opt_level, extra_args=list(extra_compile_arg))

        if not new_ir:
            new_ir_path = Path(temp_dir) / "new.ll"
            compile_to_ir(new_path, str(new_ir_path), clang_path=clang_path, opt_level=opt_level, extra_args=list(extra_compile_arg))

        old_text = Path(old_ir_path).read_text(encoding="utf-8")
        new_text = Path(new_ir_path).read_text(encoding="utf-8")
        old_norm = normalize_ir_text(old_text)
        new_norm = normalize_ir_text(new_text)
        old_funcs = parse_functions(old_norm)
        new_funcs = parse_functions(new_norm)
        diff = diff_functions(old_funcs, new_funcs)

        _render_report(old_path, new_path, diff, old_funcs, new_funcs, report_detail)


def _render_report(
    old_path: str,
    new_path: str,
    diff_result,
    old_funcs,
    new_funcs,
    report_detail: str,
) -> None:
    console.print(Panel(f"Semantic IR Diff\n[bold]old:[/bold] {old_path}\n[bold]new:[/bold] {new_path}"))

    summary_table = Table(show_header=True, header_style="bold magenta")
    summary_table.add_column("Category")
    summary_table.add_column("Count", justify="right")
    summary_table.add_row("Added functions", str(len(diff_result.added_functions)))
    summary_table.add_row("Removed functions", str(len(diff_result.removed_functions)))
    summary_table.add_row("Changed functions", str(len(diff_result.changed_functions)))
    console.print(summary_table)

    if report_detail in {"functions", "all"}:
        _render_function_diffs(diff_result, old_funcs, new_funcs)

    if report_detail == "all":
        _render_debug_stats(old_funcs, new_funcs)


def _render_function_diffs(diff_result, old_funcs, new_funcs) -> None:
    if diff_result.added_functions:
        console.print(Panel("[green]Added functions[/green]\n" + "\n".join(diff_result.added_functions), title="New Functions"))
    if diff_result.removed_functions:
        console.print(Panel("[red]Removed functions[/red]\n" + "\n".join(diff_result.removed_functions), title="Removed Functions"))

    for function_diff in diff_result.changed_functions:
        old_func = old_funcs[function_diff.name]
        new_func = new_funcs[function_diff.name]
        categories = classify_function_diff(old_func, new_func, function_diff)

        block = Panel(
            f"[bold]{function_diff.name}[/bold]\n"
            + "\n".join(function_diff.facts)
            + "\n" + "\n".join(f"[yellow]{item}[/yellow]" for item in categories),
            title=f"Function changed: {function_diff.name}",
            expand=False,
        )
        console.print(block)


def _render_debug_stats(old_funcs, new_funcs) -> None:
    old_total = sum(fn.instr_count for fn in old_funcs.values())
    new_total = sum(fn.instr_count for fn in new_funcs.values())
    delta = new_total - old_total
    stats = Table(show_header=True, header_style="bold cyan")
    stats.add_column("Metric")
    stats.add_column("Old", justify="right")
    stats.add_column("New", justify="right")
    stats.add_column("Delta", justify="right")
    stats.add_row("Total instructions", str(old_total), str(new_total), str(delta))
    stats.add_row("Functions", str(len(old_funcs)), str(len(new_funcs)), str(len(new_funcs) - len(old_funcs)))
    console.print(stats)


if __name__ == "__main__":
    main()
