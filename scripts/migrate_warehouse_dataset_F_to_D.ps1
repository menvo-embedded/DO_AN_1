# Copy warehouse AI dataset folders from external drive F: to internal drive D:.
# This script never deletes data from F:.

$ErrorActionPreference = "Stop"

$SourceRoot = "F:\warehouse_dataset"
$TargetRoot = "D:\warehouse_dataset"
$LogRoot = Join-Path $TargetRoot "_migration_logs"
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"

$Folders = @(
    "dataset_crops",
    "raw_videos_full",
    "review_crops_reid_yoloseg",
    "raw_videos_from_imou_sd"
)

New-Item -ItemType Directory -Path $TargetRoot -Force | Out-Null
New-Item -ItemType Directory -Path $LogRoot -Force | Out-Null

Write-Host "Warehouse dataset migration: F: -> D:"
Write-Host "Source root: $SourceRoot"
Write-Host "Target root: $TargetRoot"
Write-Host "Log root   : $LogRoot"
Write-Host ""

foreach ($Folder in $Folders) {
    $Source = Join-Path $SourceRoot $Folder
    $Target = Join-Path $TargetRoot $Folder
    $LogFile = Join-Path $LogRoot "$Folder-$Timestamp.log"

    if (-not (Test-Path -LiteralPath $Source)) {
        Write-Host "[SKIP] Source not found: $Source"
        continue
    }

    New-Item -ItemType Directory -Path $Target -Force | Out-Null

    Write-Host "[COPY] $Source -> $Target"
    robocopy $Source $Target /E /Z /R:2 /W:5 /MT:8 /NP /TEE /LOG:$LogFile
    $ExitCode = $LASTEXITCODE

    if ($ExitCode -le 7) {
        Write-Host "[OK] robocopy completed for $Folder with code $ExitCode"
    } else {
        throw "robocopy failed for $Folder with code $ExitCode. See log: $LogFile"
    }
}

Write-Host ""
Write-Host "Migration copy finished. Data on F: was not deleted."
Write-Host ""
Write-Host "Verify next:"
Write-Host "  .\scripts\verify_dataset_paths.ps1"
Write-Host ""
Write-Host "Then set python_cv\.env paths if needed:"
Write-Host "  WAREHOUSE_DATASET_ROOT=D:/warehouse_dataset"
Write-Host "  DATASET_CROPS_ROOT=D:/warehouse_dataset/dataset_crops"
Write-Host "  RAW_VIDEO_ROOT=D:/warehouse_dataset/raw_videos_full"
Write-Host "  REVIEW_CROP_ROOT=D:/warehouse_dataset/review_crops_reid_yoloseg"
Write-Host "  IMOU_SD_VIDEO_ROOT=D:/warehouse_dataset/raw_videos_from_imou_sd"
