param(
    [string]$RuntimeRoot = "D:\warehouse_runtime",
    [bool]$PersistUserEnv = $true,
    [bool]$CreateInsightFaceJunction = $true
)

$ErrorActionPreference = "Stop"

function Ensure-Directory {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        New-Item -ItemType Directory -Path $Path -Force | Out-Null
    }
}

function Set-EnvPath {
    param(
        [string]$Name,
        [string]$Value
    )

    Ensure-Directory $Value
    [Environment]::SetEnvironmentVariable($Name, $Value, "Process")

    if ($PersistUserEnv) {
        [Environment]::SetEnvironmentVariable($Name, $Value, "User")
    }

    Write-Host "[OK] $Name=$Value"
}

function New-Junction {
    param(
        [string]$LinkPath,
        [string]$TargetPath
    )

    Ensure-Directory $TargetPath

    if (Test-Path -LiteralPath $LinkPath) {
        $item = Get-Item -LiteralPath $LinkPath -Force

        if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            Write-Host "[OK] Junction/symlink already exists: $LinkPath"
            return
        }

        $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
        $backupPath = "$LinkPath.backup_$timestamp"
        Rename-Item -LiteralPath $LinkPath -NewName (Split-Path $backupPath -Leaf)
        Write-Host "[OK] Existing cache renamed, not deleted: $backupPath"
    }

    cmd /c mklink /J "$LinkPath" "$TargetPath" | Out-Null
    Write-Host "[OK] Junction created: $LinkPath -> $TargetPath"
}

Write-Host "Warehouse runtime migration target: $RuntimeRoot"
Write-Host "No cache/data deletion will be performed."

$paths = [ordered]@{
    TORCH_HOME               = Join-Path $RuntimeRoot "torch"
    PIP_CACHE_DIR            = Join-Path $RuntimeRoot "pip-cache"
    TEMP                     = Join-Path $RuntimeRoot "temp"
    TMP                      = Join-Path $RuntimeRoot "temp"
    YOLO_CONFIG_DIR          = Join-Path $RuntimeRoot "ultralytics"
    ULTRALYTICS_CONFIG_DIR   = Join-Path $RuntimeRoot "ultralytics"
    CONDA_PKGS_DIRS          = Join-Path $RuntimeRoot "conda-pkgs"
}

foreach ($entry in $paths.GetEnumerator()) {
    Set-EnvPath -Name $entry.Key -Value $entry.Value
}

if ($CreateInsightFaceJunction) {
    $userProfile = [Environment]::GetFolderPath("UserProfile")
    $oldInsightFace = Join-Path $userProfile ".insightface"
    $newInsightFace = Join-Path $RuntimeRoot ".insightface"

    New-Junction -LinkPath $oldInsightFace -TargetPath $newInsightFace
}

Write-Host ""
Write-Host "Done. Restart PowerShell/IDE before running the project so User environment variables are reloaded."
Write-Host "Then run: .\verify_after_migration.ps1"
