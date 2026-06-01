from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional


class ClangNotFound(RuntimeError):
    pass


def _find_executable(names: Iterable[str]) -> Optional[str]:
    for name in names:
        path = shutil.which(name)
        if path:
            return path
    return None


def find_clang(preferred: Optional[str] = None) -> str:
    candidates = []
    if preferred:
        candidates.append(preferred)
    candidates.extend(["clang", "clang.exe", "clang++", "clang++.exe"])

    repo_root = Path(__file__).resolve().parents[1]
    local_clang = repo_root / "tools" / "llvm" / "bin"
    if local_clang.exists():
        candidates.append(str(local_clang / "clang"))
        candidates.append(str(local_clang / "clang.exe"))
        candidates.append(str(local_clang / "clang++"))
        candidates.append(str(local_clang / "clang++.exe"))

    path = _find_executable(candidates)
    if not path:
        raise ClangNotFound(
            "Could not find clang/clang++ on PATH or in tools/llvm/bin. "
            "Install LLVM or specify --clang-path."
        )
    return path


def compile_to_ir(
    source_path: str,
    output_path: str,
    clang_path: Optional[str] = None,
    opt_level: str = "-O3",
    extra_args: Optional[list[str]] = None,
) -> str:
    source = Path(source_path)
    output = Path(output_path)
    suffix = source.suffix.lower()
    if suffix == ".ll":
        shutil.copyfile(source, output)
        return str(output)

    if suffix == ".bc":
        llvm_dis = _find_executable(["llvm-dis", "llvm-dis.exe"])
        if not llvm_dis:
            raise RuntimeError(
                "LLVM bitcode input requires llvm-dis on PATH to convert to textual IR."
            )
        subprocess.run([llvm_dis, "-o", str(output), str(source)], check=True)
        return str(output)

    if suffix in {".c", ".cc", ".cpp", ".cxx", ".m", ".mm"}:
        if clang_path:
            clang = Path(clang_path)
            if not clang.exists():
                raise ClangNotFound(
                    f"Could not find clang at the provided path: {clang_path}"
                )
            clang = str(clang)
        else:
            clang = find_clang("clang++" if suffix in {".cc", ".cpp", ".cxx", ".mm"} else "clang")
        command = [clang, "-S", "-emit-llvm", opt_level, "-o", str(output), str(source)]
        if extra_args:
            command.extend(extra_args)
        subprocess.run(command, check=True)
        return str(output)

    raise ValueError(
        f"Unsupported source extension '{suffix}'. Provide .c, .cpp, .ll, or .bc file."
    )
