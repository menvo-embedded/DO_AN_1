import sys, re, time, shutil, argparse, subprocess
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config.settings import CAM_ZONE2_RTSP, CAM_ZONE3_RTSP, RAW_VIDEO_ROOT

OUTPUT_ROOT = RAW_VIDEO_ROOT
FORCE_SUBTYPE = 0

def force_subtype(url, subtype):
    if "subtype=" in url:
        return re.sub(r"subtype=\d+", f"subtype={subtype}", url)
    if "?" in url:
        return url + f"&subtype={subtype}"
    return url + f"?subtype={subtype}"

def mask_rtsp(url):
    return re.sub(r"://([^:/]+):([^@]+)@", r"://\1:***@", url)

def find_ffmpeg():
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    winget_root = Path.home() / "AppData" / "Local" / "Microsoft" / "WinGet" / "Packages"
    if winget_root.exists():
        found = list(winget_root.rglob("ffmpeg.exe"))
        if found:
            return str(found[0])
    raise RuntimeError("Không tìm thấy ffmpeg.exe")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zone", required=True, choices=["zone2", "zone3"])
    parser.add_argument("--seconds", type=int, default=18000)
    parser.add_argument("--session", default="raw_session")
    args = parser.parse_args()

    zone = args.zone
    seconds = int(args.seconds)
    session = args.session.strip().replace(" ", "_")

    base_url = CAM_ZONE2_RTSP if zone == "zone2" else CAM_ZONE3_RTSP
    rtsp_url = force_subtype(base_url, FORCE_SUBTYPE)

    out_dir = OUTPUT_ROOT / session
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{zone}_raw_stable_mainstream_{ts}_{seconds}s.mkv"

    ffmpeg = find_ffmpeg()

    cmd = [
        ffmpeg,
        "-y",
        "-rtsp_transport", "tcp",
        "-i", rtsp_url,
        "-t", str(seconds),
        "-map", "0:v:0",
        "-c:v", "copy",
        "-an",
        str(out_path),
    ]

    print("========== RAW STABLE RECORD ==========")
    print("Zone   :", zone)
    print("RTSP   :", mask_rtsp(rtsp_url))
    print("Output :", out_path)
    print("Mode   : stream copy, no preview, no crop, no detect")
    print("Stop   : nhấn q trong terminal FFmpeg")
    print("======================================")

    process = subprocess.run(cmd)

    print("[DONE]", out_path)
    print("returncode =", process.returncode)

if __name__ == "__main__":
    main()
