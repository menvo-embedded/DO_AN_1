from pathlib import Path
import torch

src = Path("models/reid_resnet50_v3_cleanval.pth")
backup = Path("models/reid_resnet50_v3_cleanval_full.pth")
dst = Path("models/reid_resnet50_v3_engine.pth")

if not src.exists():
    raise FileNotFoundError(src)

# backup model gốc
if not backup.exists():
    backup.write_bytes(src.read_bytes())

state = torch.load(src, map_location="cpu")

if isinstance(state, dict):
    if "state_dict" in state:
        state = state["state_dict"]
    elif "model_state_dict" in state:
        state = state["model_state_dict"]

new_state = {}

for k, v in state.items():
    # model Kaggle: backbone.xxx
    # engine local cũ: xxx
    if k.startswith("backbone."):
        nk = k.replace("backbone.", "", 1)

        # bỏ classifier nếu có
        if nk.startswith("fc."):
            continue

        new_state[nk] = v

torch.save(new_state, dst)

print("DONE")
print("Input :", src)
print("Backup:", backup)
print("Output:", dst)
print("Keys  :", len(new_state))
