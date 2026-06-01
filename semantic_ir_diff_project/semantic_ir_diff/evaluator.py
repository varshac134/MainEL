from __future__ import annotations

import csv
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import click
from rich.console import Console
from rich.table import Table

from .compiler import compile_to_ir
from .classify import classify_function_diff
from .diff import diff_functions
from .normalize import normalize_ir_text
from .parser import parse_functions

console = Console()


@dataclass
class EvaluationCase:
    case_id: str
    old_commit: str
    new_commit: str
    file_path: str
    description: str
    expected_labels: List[str]


def _normalize_labels(text: str) -> List[str]:
    return [item.strip().lower() for item in re.split(r"[;,\\|]", text) if item.strip()]


def _read_csv_manifest(manifest_path: Path) -> List[EvaluationCase]:
    cases: List[EvaluationCase] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader, start=1):
            cases.append(
                EvaluationCase(
                    case_id=str(row.get("id", str(index))).strip(),
                    old_commit=row["old_commit"].strip(),
                    new_commit=row["new_commit"].strip(),
                    file_path=row["file_path"].strip(),
                    description=row.get("description", "").strip(),
                    expected_labels=_normalize_labels(row.get("expected_labels", "")),
                )
            )
    return cases


def _git_show(repo_path: Path, commit: str, file_path: str, output: Path) -> None:
    result = subprocess.run(
        ["git", "-C", str(repo_path), "show", f"{commit}:{file_path}"],
        capture_output=True,
        text=True,
        check=True,
    )
    output.write_text(result.stdout, encoding="utf-8")


def _analyze_pair(
    repo_path: Path,
    case: EvaluationCase,
    clang_path: Optional[str],
    opt_level: str,
    extra_args: List[str],
) -> dict:
    with tempfile.TemporaryDirectory() as temp_dir:
        tmp = Path(temp_dir)
        old_source = tmp / "old_source"
        new_source = tmp / "new_source"
        _git_show(repo_path, case.old_commit, case.file_path, old_source)
        _git_show(repo_path, case.new_commit, case.file_path, new_source)

        old_ir = tmp / "old.ll"
        new_ir = tmp / "new.ll"
        compile_to_ir(str(old_source), str(old_ir), clang_path=clang_path, opt_level=opt_level, extra_args=extra_args)
        compile_to_ir(str(new_source), str(new_ir), clang_path=clang_path, opt_level=opt_level, extra_args=extra_args)

        old_text = old_ir.read_text(encoding="utf-8")
        new_text = new_ir.read_text(encoding="utf-8")
        old_norm = normalize_ir_text(old_text)
        new_norm = normalize_ir_text(new_text)
        old_funcs = parse_functions(old_norm)
        new_funcs = parse_functions(new_norm)
        diff = diff_functions(old_funcs, new_funcs)

        categories: List[str] = []
        for function_diff in diff.changed_functions:
            old_func = old_funcs[function_diff.name]
            new_func = new_funcs[function_diff.name]
            categories.extend(classify_function_diff(old_func, new_func, function_diff))

        categories.extend([f"added function: {name}" for name in diff.added_functions])
        categories.extend([f"removed function: {name}" for name in diff.removed_functions])
        categories = sorted(set(categories))

        description_lower = case.description.lower()
        description_hits = [cat for cat in categories if any(word in description_lower for word in re.findall(r"\w+", cat.lower()))]

        return {
            "case_id": case.case_id,
            "file_path": case.file_path,
            "old_commit": case.old_commit,
            "new_commit": case.new_commit,
            "description": case.description,
            "expected_labels": case.expected_labels,
            "categories": categories,
            "description_matches": description_hits,
            "missing_expected": [label for label in case.expected_labels if label not in categories],
            "unexpected": [label for label in categories if label not in case.expected_labels and case.expected_labels],
        }


def _render_evaluation_summary(results: List[dict]) -> None:
    summary = Table(show_header=True, header_style="bold blue")
    summary.add_column("Case")
    summary.add_column("File")
    summary.add_column("Changed funcs", justify="right")
    summary.add_column("Categories")
    summary.add_column("Description hits")
    summary.add_column("Missing expected")

    for result in results:
        summary.add_row(
            result["case_id"],
            result["file_path"],
            str(len(result["categories"])),
            ", ".join(result["categories"][:3]) + ("..." if len(result["categories"]) > 3 else ""),
            ", ".join(result["description_matches"][:2]) or "none",
            ", ".join(result["missing_expected"][:2]) or "none",
        )

    console.print(Panel(summary, title="Evaluation Summary", expand=False))


def _render_case_details(result: dict) -> None:
    console.print(Panel(f"[bold]Case {result['case_id']}[/bold]\n{result['file_path']}\n{result['old_commit']} -> {result['new_commit']}"))
    if result["categories"]:
        console.print("[green]Detected semantic categories:[/green] " + ", ".join(result["categories"]))
    else:
        console.print("[yellow]No semantic categories detected.[/yellow]")
    if result["description_matches"]:
        console.print("[cyan]Matches in commit description:[/cyan] " + ", ".join(result["description_matches"]))
    if result["missing_expected"]:
        console.print("[red]Missing expected labels:[/red] " + ", ".join(result["missing_expected"]))


def _load_manifest(manifest_path: Path) -> List[EvaluationCase]:
    if manifest_path.suffix.lower() == ".csv":
        return _read_csv_manifest(manifest_path)
    raise click.BadParameter("Manifest must be a .csv file with headers: id,old_commit,new_commit,file_path,description,expected_labels")


@click.command()
@click.option("--repo-path", type=click.Path(exists=True, file_okay=False), default=".", help="Path to the git repository containing revisions.")
@click.option("--manifest", type=click.Path(exists=True, dir_okay=False), required=True, help="CSV manifest describing evaluation cases.")
@click.option("--clang-path", type=click.Path(exists=True), default=None, help="Explicit clang/clang++ executable path.")
@click.option("--opt-level", default="-O3", help="Optimization level for LLVM IR emission.")
@click.option("--extra-compile-arg", multiple=True, help="Extra compile arguments to pass to clang.")
@click.option("--detail/--no-detail", default=False, help="Render per-case category details.")
def main(
    repo_path: str,
    manifest: str,
    clang_path: Optional[str],
    opt_level: str,
    extra_compile_arg: tuple[str, ...],
    detail: bool,
) -> None:
    """Run semantic diff evaluation across a manifest of commit pairs."""
    cases = _load_manifest(Path(manifest))
    results: List[dict] = []
    for case in cases:
        console.print(f"[bold]Evaluating case {case.case_id} ({case.file_path})[/bold]")
        try:
            result = _analyze_pair(Path(repo_path), case, clang_path, opt_level, list(extra_compile_arg))
            results.append(result)
            if detail:
                _render_case_details(result)
        except subprocess.CalledProcessError as exc:
            console.print(f"[red]Git or compile error for case {case.case_id}: {exc}[/red]")

    _render_evaluation_summary(results)


if __name__ == "__main__":
    main()
