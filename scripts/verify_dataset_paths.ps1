# Verify expected local dataset paths after migration to D:\warehouse_dataset.

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$DatasetRoot = "D:\warehouse_dataset"
$DatasetCrops = Join-Path $DatasetRoot "dataset_crops"
$RawVideos = Join-Path $DatasetRoot "raw_videos_full"
$ReviewCrops = Join-Path $DatasetRoot "review_crops_reid_yoloseg"
$GalleryFile = Join-Path $RepoRoot "python_cv\data\gallery\gallery.pkl"

$RequiredEmployees = @("NV001", "NV002", "NV003", "NV004", "NV005")
$VideoExts = @("*.mp4", "*.mkv", "*.avi", "*.mov")
$ImageExts = @("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")

function Count-Files($Path, $Patterns) {
    if (-not (Test-Path -LiteralPath $Path)) {
        return 0
    }

    $Count = 0
    foreach ($Pattern in $Patterns) {
        $Count += @(Get-ChildItem -LiteralPath $Path -Recurse -File -Filter $Pattern -ErrorAction SilentlyContinue).Count
    }
    return $Count
}

function Print-Status($Label, $Ok, $Detail) {
    if ($Ok) {
        Write-Host "[OK]   $Label - $Detail"
    } else {
        Write-Host "[WARN] $Label - $Detail"
    }
}

Write-Host "Warehouse dataset path verification"
Write-Host "Repo root   : $RepoRoot"
Write-Host "Dataset root: $DatasetRoot"
Write-Host ""

$RootExists = Test-Path -LiteralPath $DatasetRoot
Print-Status "Dataset root" $RootExists $DatasetRoot

$CropsExists = Test-Path -LiteralPath $DatasetCrops
Print-Status "dataset_crops" $CropsExists $DatasetCrops

foreach ($EmployeeId in $RequiredEmployees) {
    $Folder = Join-Path $DatasetCrops $EmployeeId
    $Exists = Test-Path -LiteralPath $Folder
    $ImageCount = Count-Files $Folder $ImageExts
    Print-Status "dataset_crops\$EmployeeId" ($Exists -and $ImageCount -gt 0) "$ImageCount image files"
}

$RawVideoCount = Count-Files $RawVideos $VideoExts
Print-Status "raw_videos_full" ($RawVideoCount -gt 0) "$RawVideoCount video files"

$ReviewCropCount = Count-Files $ReviewCrops $ImageExts
Print-Status "review_crops_reid_yoloseg" ($ReviewCropCount -gt 0) "$ReviewCropCount image files"

$GalleryExists = Test-Path -LiteralPath $GalleryFile
Print-Status "gallery.pkl" $GalleryExists $GalleryFile

Write-Host ""
Write-Host "Summary:"
Write-Host "  Dataset root exists       : $RootExists"
Write-Host "  Employee folders checked  : $($RequiredEmployees -join ', ')"
Write-Host "  Raw video files           : $RawVideoCount"
Write-Host "  Review crop image files   : $ReviewCropCount"
Write-Host "  Body gallery exists       : $GalleryExists"
Write-Host ""
Write-Host "If some warnings are expected on a clean submission machine, the demo can still run when the required model files and python_cv\data\gallery\gallery.pkl are present."
