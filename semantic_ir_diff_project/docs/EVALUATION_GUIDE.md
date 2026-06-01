# Semantic IR Diff Evaluation Guide

This guide explains how to run the semantic IR diff evaluation harness on a set of real Git commit pairs.

## Purpose

The evaluation harness is designed to help you compare semantic changes detected by the tool against commit descriptions or expected labels. It is especially useful for performance-sensitive codebases and LLVM optimization research.

## Manifest Format

Create a CSV file with the following headers:

- `id`: Optional unique identifier for the case
- `old_commit`: Git object name for the old revision
- `new_commit`: Git object name for the new revision
- `file_path`: Path to the source file inside the repository
- `description`: Commit message or description of the code change
- `expected_labels`: Optional semicolon-separated expected semantic categories

Example:

```csv
id,old_commit,new_commit,file_path,description,expected_labels
1,abc123,def456,src/main.c,"Change loop bound to runtime variable","loop vectorization lost;instruction count changed"
2,789abc,012def,src/foo.cpp,"Inline helper into hot path","calls removed;inlining changed"
```

## Running Evaluation

From the repository root:

```bash
python -m semantic_ir_diff.evaluator --repo-path . --manifest docs/evaluation_manifest.csv --detail
```

If the package is installed in editable mode, use the script:

```bash
semantic-ir-diff-eval --repo-path . --manifest docs/evaluation_manifest.csv --detail
```

## Output

- A summary table of evaluation cases
- Optional per-case details with detected semantic categories
- Information on category matches between the tool and the commit description
- Missing expected labels when provided in the manifest

## Notes

- The harness uses `git show` to extract file contents from the specified commits.
- It compiles revisions to LLVM IR with `clang` at the selected optimization level.
- Install `clang` locally using `scripts/install_clang.ps1` on Windows or `scripts/install_clang.sh` on Linux/macOS.
- The tool also searches `tools/llvm/bin` for a local clang installation.
- You can extend the manifest with real LLVM repo commits or application commits.
- For best results, use source files in C/C++ or already-built LLVM IR.
