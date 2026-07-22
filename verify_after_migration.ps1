param(
    [string]$ProjectRoot = "D:\warehouse-access-rfid-cv",
    [string]$PythonExe = "D:\UV4\anaconda3\python.exe",
    [int]$CameraTimeoutSeconds = 8
)

$ErrorActionPreference = "Stop"

function Test-TcpPort {
    param(
        [string]$HostName,
        [int]$Port,
        [int]$TimeoutMs = 3000
    )

    $client = [System.Net.Sockets.TcpClient]::new()
    try {
        $iar = $client.BeginConnect($HostName, $Port, $null, $null)
        $ok = $iar.AsyncWaitHandle.WaitOne($TimeoutMs, $false)
        if ($ok -and $client.Connected) {
            Write-Host "[OK] TCP $HostName`:$Port open"
            return $true
        }

        Write-Host "[FAIL] TCP $HostName`:$Port closed or timed out"
        return $false
    }
    finally {
        $client.Close()
    }
}

if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Python executable not found: $PythonExe"
}

$pythonCvRoot = Join-Path $ProjectRoot "python_cv"
if (-not (Test-Path -LiteralPath $pythonCvRoot)) {
    throw "python_cv folder not found: $pythonCvRoot"
}

$env:PROJECT_ROOT = $ProjectRoot
$env:PYTHONPATH = "$pythonCvRoot;$env:PYTHONPATH"

Write-Host "== Environment variables =="
$names = @(
    "TORCH_HOME",
    "PIP_CACHE_DIR",
    "TEMP",
    "TMP",
    "YOLO_CONFIG_DIR",
    "ULTRALYTICS_CONFIG_DIR",
    "CONDA_PKGS_DIRS"
)

foreach ($name in $names) {
    $value = [Environment]::GetEnvironmentVariable($name, "Process")
    if (-not $value) {
        $value = [Environment]::GetEnvironmentVariable($name, "User")
    }
    if ($value) {
        Write-Host "[OK] $name=$value"
    }
    else {
        Write-Host "[WARN] $name is not set"
    }
}

Write-Host ""
Write-Host "== Python / CV / AI checks =="

$code = @'
import os
import socket
import sys
from pathlib import Path
from urllib.parse import urlparse

project_root = Path(os.environ["PROJECT_ROOT"])
python_cv = project_root / "python_cv"
sys.path.insert(0, str(python_cv))
os.chdir(str(python_cv))

def report(name, ok, detail=""):
    status = "OK" if ok else "FAIL"
    print(f"[{status}] {name}{(': ' + detail) if detail else ''}")

try:
    import cv2
    report("import cv2", True, getattr(cv2, "__version__", "unknown"))
except Exception as exc:
    report("import cv2", False, repr(exc))
    raise

try:
    import torch
    report("import torch", True, getattr(torch, "__version__", "unknown"))
    report("torch.cuda.is_available()", bool(torch.cuda.is_available()), str(torch.cuda.is_available()))
except Exception as exc:
    report("import torch", False, repr(exc))
    raise

try:
    import ultralytics
    report("import ultralytics", True, getattr(ultralytics, "__version__", "unknown"))
except Exception as exc:
    report("import ultralytics", False, repr(exc))
    raise

try:
    import insightface
    report("import insightface", True, getattr(insightface, "__version__", "unknown"))
except Exception as exc:
    report("import insightface", False, repr(exc))
    raise

try:
    from config.settings import (
        ENABLE_ZONE1,
        ENABLE_ZONE2,
        ENABLE_MQTT,
        ENABLE_INSIGHTFACE,
        CAM_ZONE1_INDEX,
        CAM_ZONE2_TYPE,
        CAM_ZONE2_INDEX,
        CAM_ZONE2_RTSP,
        MQTT_BROKER,
        MQTT_PORT,
        MQTT_TOPIC,
    )
    parsed = urlparse(CAM_ZONE2_RTSP) if CAM_ZONE2_RTSP else None
    zone2_host = parsed.hostname if parsed else ""
    report(".env load OK", True, f"zone1={ENABLE_ZONE1}, zone2={ENABLE_ZONE2}, mqtt={ENABLE_MQTT}, insightface={ENABLE_INSIGHTFACE}")
    report("Zone 1 config", True, f"index={CAM_ZONE1_INDEX}")
    report("Zone 2 config", True, f"type={CAM_ZONE2_TYPE}, index={CAM_ZONE2_INDEX}, rtsp_host={zone2_host}")
    report("MQTT config", True, f"{MQTT_BROKER}:{MQTT_PORT}, topic={MQTT_TOPIC}")
except Exception as exc:
    report(".env load OK", False, repr(exc))
    raise

try:
    cap1 = cv2.VideoCapture(CAM_ZONE1_INDEX, cv2.CAP_DSHOW)
    if not cap1.isOpened():
        cap1.release()
        cap1 = cv2.VideoCapture(CAM_ZONE1_INDEX)
    ok1, frame1 = cap1.read() if cap1.isOpened() else (False, None)
    shape1 = None if frame1 is None else tuple(frame1.shape)
    report("camera Zone 1 read frame", bool(ok1 and frame1 is not None), f"index={CAM_ZONE1_INDEX}, shape={shape1}")
finally:
    try:
        cap1.release()
    except Exception:
        pass

try:
    if CAM_ZONE2_TYPE != "rtsp":
        report("camera Zone 2 RTSP read frame", False, f"CAM_ZONE2_TYPE={CAM_ZONE2_TYPE}, expected rtsp")
    else:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        cap2 = cv2.VideoCapture()
        cap2.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
        cap2.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
        opened = cap2.open(CAM_ZONE2_RTSP, cv2.CAP_FFMPEG)
        ok2, frame2 = cap2.read() if opened and cap2.isOpened() else (False, None)
        shape2 = None if frame2 is None else tuple(frame2.shape)
        parsed = urlparse(CAM_ZONE2_RTSP)
        report("camera Zone 2 RTSP read frame", bool(ok2 and frame2 is not None), f"host={parsed.hostname}, shape={shape2}")
finally:
    try:
        cap2.release()
    except Exception:
        pass
'@

$tempScript = Join-Path ([System.IO.Path]::GetTempPath()) "warehouse_verify_after_migration.py"
Set-Content -Path $tempScript -Value $code -Encoding UTF8
& $PythonExe $tempScript
$pythonExit = $LASTEXITCODE
Remove-Item -LiteralPath $tempScript -Force -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "== MQTT port check =="

Push-Location $pythonCvRoot
try {
    $mqttInfo = & $PythonExe -c "from config.settings import MQTT_BROKER, MQTT_PORT; print(f'{MQTT_BROKER}|{MQTT_PORT}')"
}
finally {
    Pop-Location
}

$parts = $mqttInfo.Trim().Split("|")
if ($parts.Count -eq 2) {
    [void](Test-TcpPort -HostName $parts[0] -Port ([int]$parts[1]))
}
else {
    Write-Host "[FAIL] Could not read MQTT_BROKER/MQTT_PORT from settings"
}

if ($pythonExit -ne 0) {
    exit $pythonExit
}
