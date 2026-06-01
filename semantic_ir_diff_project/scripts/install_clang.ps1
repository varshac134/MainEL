<#
Downloads and installs a local LLVM toolchain into tools/llvm.
This script does not require admin privileges.
#>

param(
    [string]$Version = "18.0.6",
    [string]$Destination = "tools/llvm",
    [string]$ArchivePath = ""
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Path $PSScriptRoot -Parent
$destinationPath = Join-Path $root $Destination
if (-Not (Test-Path $destinationPath)) {
    New-Item -ItemType Directory -Path $destinationPath | Out-Null
}

$candidates = @(
    "clang+llvm-$Version-x86_64-windows-msvc.zip",
    "clang+llvm-$Version-x86_64-windows-gnu.zip",
    "LLVM-$Version-win64.zip"
)

if ($ArchivePath -and (Test-Path $ArchivePath)) {
    Write-Host "Using pre-downloaded archive: $ArchivePath"
    $archivePath = $ArchivePath
} elseif ($ArchivePath) {
    throw "Archive path '$ArchivePath' does not exist."
} else {
    $archivePath = $null
    foreach ($archiveName in $candidates) {
        $url = "https://github.com/llvm/llvm-project/releases/download/llvmorg-$Version/$archiveName"
        Write-Host "Trying LLVM download: $url"
        try {
            $destinationArchive = Join-Path $destinationPath $archiveName
            Invoke-WebRequest -Uri $url -OutFile $destinationArchive -UseBasicParsing
            $archivePath = $destinationArchive
            break
        } catch {
            $errorMessage = $_.Exception.Message
            Write-Host ("Failed to download " + $archiveName + ": " + $errorMessage)
        }
    }

    if (-not $archivePath) {
        Write-Error "Could not download LLVM automatically."
        Write-Host "Please download one of the following from GitHub manually:"
        foreach ($archiveName in $candidates) {
            Write-Host "  https://github.com/llvm/llvm-project/releases/download/llvmorg-$Version/$archiveName"
        }
        Write-Host "Then pass the archive path to this script using -ArchivePath."
        Write-Host "Example: .\scripts\install_clang.ps1 -ArchivePath .\tools\llvm\clang+llvm-18.0.6-x86_64-windows-msvc.zip"
        exit 1
    }
}

Write-Host "Extracting to $destinationPath ..."
Expand-Archive -LiteralPath $archivePath -DestinationPath $destinationPath -Force

$extractedFolder = Get-ChildItem -Path $destinationPath -Directory | Where-Object { $_.Name -like 'LLVM-*' } | Select-Object -First 1
if ($null -eq $extractedFolder) {
    throw "Could not find extracted LLVM folder under $destinationPath"
}

$targetBin = Join-Path $destinationPath "bin"
if (-Not (Test-Path $targetBin)) {
    New-Item -ItemType Directory -Path $targetBin | Out-Null
}

Write-Host "Moving LLVM binaries to $targetBin ..."
Move-Item -Path (Join-Path $extractedFolder.FullName 'bin\*') -Destination $targetBin

Remove-Item $archivePath
Remove-Item -Path $extractedFolder.FullName -Recurse -Force

Write-Host "LLVM installed to $targetBin"
Write-Host "Use clang from $targetBin\clang.exe or pass --clang-path to the CLI."
