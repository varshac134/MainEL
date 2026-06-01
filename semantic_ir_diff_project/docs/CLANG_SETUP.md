# LLVM/Clang Setup for Semantic IR Diff

This project requires `clang` or `clang++` for compiling source code into LLVM IR.
The semantic diff tool will also look for a local `clang` installation under `tools/llvm/bin`.

## Install LLVM locally

### Windows

Run the PowerShell installer script:

```powershell
cd path\to\model_release
.\scripts\install_clang.ps1
```

This downloads a prebuilt LLVM release and extracts it under `tools/llvm`.
After extraction, update the repo if necessary:

```powershell
Move-Item tools\llvm\LLVM-18.0.6\bin tools\llvm\bin
```

### Linux / macOS

Run the shell installer script:

```bash
cd /path/to/model_release
./scripts/install_clang.sh
```

After extraction, you may need to move the LLVM binaries into `tools/llvm/bin`:

```bash
mv tools/llvm/LLVM-18.0.6/bin tools/llvm/bin
```

## Using local clang automatically

The semantic diff tool searches the system `PATH` first, then `tools/llvm/bin` relative to the repository root.
If you install LLVM locally using one of the scripts above, the tool should find it automatically.

If the script fails because the asset is not available or your network blocks GitHub, download one of these archives manually:

- `clang+llvm-18.0.6-x86_64-windows-msvc.zip`
- `clang+llvm-18.0.6-x86_64-windows-gnu.zip`
- `clang+llvm-18.0.6-x86_64-linux-gnu.tar.xz`
- `clang+llvm-18.0.6-x86_64-linux-musl.tar.xz`

Then extract the file into `tools/llvm` so that `tools/llvm/bin/clang` exists.

### Use a pre-downloaded archive

Windows:
```powershell
.\scripts\install_clang.ps1 -ArchivePath .\tools\llvm\clang+llvm-18.0.6-x86_64-windows-msvc.zip
```

Linux/macOS:
```bash
./scripts/install_clang.sh /path/to/clang+llvm-18.0.6-x86_64-linux-gnu.tar.xz
```

## Manual clang path

If you prefer, pass the clang executable path directly:

```bash
semantic-ir-diff old.c new.c --clang-path tools/llvm/bin/clang
```

## Troubleshooting

- If the tool still cannot find `clang`, ensure the binary exists in `tools/llvm/bin`.
- On Windows, ensure the downloaded LLVM archive contains `clang.exe` in `tools/llvm/bin`.
- If you installed LLVM in a different location, use `--clang-path`.
