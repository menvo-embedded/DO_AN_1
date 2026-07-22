# Cleanup commands draft for Warehouse Access RFID + CV.
# Review this file before running. By default, commands run with WhatIf enabled.
# Set $WhatIfPreference = $false only after confirming the archive/delete plan.

$ErrorActionPreference = "Stop"
$WhatIfPreference = $true

$ArchiveRoot = "archive"
$ArchiveCode = Join-Path $ArchiveRoot "code_old"
$ArchiveDebug = Join-Path $ArchiveRoot "debug_files"
$ArchiveHardware = Join-Path $ArchiveRoot "hardware_old"
$ArchiveNested = Join-Path $ArchiveRoot "nested_stale_repo"

function Ensure-Dir($Path) {
    if (-not (Test-Path -LiteralPath $Path)) {
        New-Item -ItemType Directory -Path $Path -Force | Out-Null
    }
}

function Move-IfExists($Source, $DestinationDir) {
    if (Test-Path -LiteralPath $Source) {
        Ensure-Dir $DestinationDir
        Move-Item -LiteralPath $Source -Destination $DestinationDir -Force -WhatIf:$WhatIfPreference
    }
}

function Remove-IfExists($Path) {
    if (Test-Path -LiteralPath $Path) {
        Remove-Item -LiteralPath $Path -Recurse -Force -WhatIf:$WhatIfPreference
    }
}

Ensure-Dir $ArchiveRoot
Ensure-Dir $ArchiveCode
Ensure-Dir $ArchiveDebug
Ensure-Dir $ArchiveHardware
Ensure-Dir $ArchiveNested

# ---------------------------------------------------------------------------
# Archive stale/nested code and old experiments
# ---------------------------------------------------------------------------
Move-IfExists "warehouse-access-rfid-cv" $ArchiveNested
Move-IfExists "experiments\old_tests" $ArchiveCode

# ---------------------------------------------------------------------------
# Archive backup/copy files
# ---------------------------------------------------------------------------
Move-IfExists "python_cv\main_backup_before_disable_mqtt.py" $ArchiveCode
Move-IfExists "python_cv\main_backup_before_zone3.py" $ArchiveCode
Move-IfExists "python_cv\config\settings_backup_before_core_fix.py" $ArchiveCode
Move-IfExists "python_cv\config\settings_backup_before_disable_mqtt.py" $ArchiveCode
Move-IfExists "python_cv\config\settings_backup_before_remove_nv006_nv007.py" $ArchiveCode
Move-IfExists "python_cv\detection\camera_zone1_backup_before_core_fix.py" $ArchiveCode
Move-IfExists "python_cv\detection\camera_zone2_backup_before_core_fix.py" $ArchiveCode
Move-IfExists "python_cv\detection\camera_zone3_backup_before_core_fix.py" $ArchiveCode
Move-IfExists "python_cv\fusion\fusion_layer_backup_before_core_fix.py" $ArchiveCode
Move-IfExists "python_cv\reid\gallery_backup_before_top5_fix.py" $ArchiveCode
Move-IfExists "python_cv\reid\reid_engine_backup_before_core_fix.py" $ArchiveCode
Move-IfExists "python_cv\tools\crop_reid_from_video.py.py" $ArchiveCode

# Dashboard copy not used by python_cv/main.py; archive only after confirming
# python_cv/dashboard/app.py is the chosen dashboard.
Move-IfExists "python_cv\app.py" $ArchiveCode

# ---------------------------------------------------------------------------
# Archive debug notes and local one-off images
# ---------------------------------------------------------------------------
Move-IfExists "python_cv\code_review_core_01.txt" $ArchiveDebug
Move-IfExists "python_cv\code_review_batch_02_database_dashboard.txt" $ArchiveDebug
Move-IfExists "python_cv\project_tree.txt" $ArchiveDebug
Move-IfExists "python_cv\gallery_py_current.txt" $ArchiveDebug
Move-IfExists "python_cv\test_image.py" $ArchiveDebug
Move-IfExists "python_cv\check_dataset.py" $ArchiveDebug
Move-IfExists "python_cv\fix_db.py" $ArchiveDebug

Get-ChildItem -LiteralPath "python_cv" -File -Filter "face_test_*.jpg" -ErrorAction SilentlyContinue |
    ForEach-Object { Move-IfExists $_.FullName $ArchiveDebug }
Get-ChildItem -LiteralPath "python_cv" -File -Filter "test_crop_*.jpg" -ErrorAction SilentlyContinue |
    ForEach-Object { Move-IfExists $_.FullName $ArchiveDebug }
Move-IfExists "python_cv\test.jpg" $ArchiveDebug
Move-IfExists "python_cv\test_face.jpg" $ArchiveDebug
Move-IfExists "python_cv\test_face_face_test.jpg" $ArchiveDebug

# ---------------------------------------------------------------------------
# Archive hardware intermediate files after choosing final KiCad project
# ---------------------------------------------------------------------------
Move-IfExists "hardware\pcb\warehouse_rfid_controller\.history" $ArchiveHardware
Move-IfExists "hardware\pcb\warehouse_rfid_controller\FINAL.dsn" $ArchiveHardware
Move-IfExists "hardware\pcb\warehouse_rfid_controller\FINAL.rules" $ArchiveHardware
Move-IfExists "hardware\pcb\warehouse_rfid_controller\FINAL.ses" $ArchiveHardware
Move-IfExists "hardware\pcb\warehouse_rfid_controller\warehouse.dsn" $ArchiveHardware
Move-IfExists "hardware\pcb\warehouse_rfid_controller\warehouse.rules" $ArchiveHardware
Move-IfExists "hardware\pcb\warehouse_rfid_controller\warehouse.ses" $ArchiveHardware

# ---------------------------------------------------------------------------
# Remove generated caches/build products. Safe to regenerate.
# ---------------------------------------------------------------------------
Remove-IfExists ".pio"
Remove-IfExists "firmware\.pio"
Remove-IfExists ".pytest_cache"
Remove-IfExists ".vscode"

Get-ChildItem -Path "." -Recurse -Directory -Force -Filter "__pycache__" -ErrorAction SilentlyContinue |
    ForEach-Object { Remove-IfExists $_.FullName }
Get-ChildItem -Path "." -Recurse -File -Force -Include "*.pyc","*.pyo" -ErrorAction SilentlyContinue |
    ForEach-Object { Remove-IfExists $_.FullName }

# ---------------------------------------------------------------------------
# Do not delete sensitive/heavy data automatically. Review manually.
# ---------------------------------------------------------------------------
Write-Host "Manual review required before submission:"
Write-Host "- python_cv\.env"
Write-Host "- firmware\src\main.cpp hardcoded WiFi/MQTT secrets"
Write-Host "- python_cv\data\dataset_raw\*.mp4"
Write-Host "- python_cv\data\dataset_crops\"
Write-Host "- python_cv\data\gallery\*.pkl and python_cv\data\face_gallery\*.pkl"
Write-Host "- python_cv\outputs\"
Write-Host "- model files: *.pt, *.pth, *.onnx, *.zip"

Write-Host "Draft complete. WhatIf is currently: $WhatIfPreference"
