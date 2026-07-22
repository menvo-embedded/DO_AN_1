"""
Fine-tune OSNet x0_5 trên warehouse dataset.
Usage:
  python reid/trainer.py --stage1   # Train trên Market-1501
  python reid/trainer.py --stage2   # Fine-tune trên warehouse
  python reid/trainer.py --eval     # Evaluate model
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

import torchreid
from config.settings import DATASET_CROPS_ROOT
from utils.logger import get_logger

log = get_logger("trainer")

# ── Config ────────────────────────────────────────────────────────────────────
MARKET1501_ROOT = "data/market1501"   # Download từ: https://zheng-lab.cec.nus.edu.sg/Project/project_reid.html
WAREHOUSE_ROOT  = str(DATASET_CROPS_ROOT)
MODEL_DIR       = "models"
STAGE1_WEIGHTS  = "models/osnet_x0_5_market1501.pth"
STAGE2_WEIGHTS  = "models/osnet_x0_5_warehouse.pth"

# ── Dataset ───────────────────────────────────────────────────────────────────
class WarehouseDataset(Dataset):
    """Load ảnh từ data/dataset_crops/NV001/, NV002/, ..."""
    def __init__(self, root: str, transform=None):
        self.samples = []
        self.transform = transform
        self.class_to_idx = {}

        root_path = Path(root)
        if not root_path.exists():
            raise FileNotFoundError(f"Dataset root not found: {root}")

        classes = sorted([d.name for d in root_path.iterdir() if d.is_dir()])
        if not classes:
            raise ValueError(f"No class folders found in {root}")

        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.num_classes  = len(classes)

        for cls_name in classes:
            cls_dir = root_path / cls_name
            for img_path in cls_dir.glob("*.jpg"):
                self.samples.append((str(img_path), self.class_to_idx[cls_name]))

        log.info(f"Dataset: {len(self.samples)} images, {self.num_classes} classes")
        log.info(f"Classes: {classes}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

# ── Transforms ────────────────────────────────────────────────────────────────
TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Pad(10),
    transforms.RandomCrop((256, 128)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2,
                           saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.4)),
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ── Model builder ─────────────────────────────────────────────────────────────
def build_model(num_classes: int, pretrained_path: str = None):
    model = torchreid.models.build_model(
        name="osnet_x0_5",
        num_classes=num_classes,
        pretrained=True,   # ImageNet weights
    )
    if pretrained_path and Path(pretrained_path).exists():
        log.info(f"Loading weights: {pretrained_path}")
        state = torch.load(pretrained_path, map_location="cpu")
        # Handle different checkpoint formats
        if "state_dict" in state:
            state = state["state_dict"]
        try:
            model.load_state_dict(state, strict=False)
            log.info("Weights loaded (strict=False)")
        except Exception as e:
            log.warning(f"Partial weight load: {e}")
    return model

# ── Training loop ─────────────────────────────────────────────────────────────
def train(model, dataloader, optimizer, scaler, device, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    total_loss = 0.0
    correct    = 0
    total      = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        with autocast():
            outputs = model(images)
            loss    = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total   += labels.size(0)

        if (batch_idx + 1) % 20 == 0:
            acc = 100.0 * correct / total
            log.info(f"Epoch {epoch} [{batch_idx+1}/{len(dataloader)}] "
                     f"Loss={total_loss/(batch_idx+1):.4f} Acc={acc:.1f}%")

    return total_loss / len(dataloader), 100.0 * correct / total

# ── Evaluation ────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    all_embeds = []
    all_labels = []

    for images, labels in dataloader:
        images = images.to(device)
        feats  = model(images)
        feats  = feats / feats.norm(dim=1, keepdim=True)
        all_embeds.append(feats.cpu())
        all_labels.extend(labels.numpy())

    embeds = torch.cat(all_embeds, dim=0).numpy()
    labels = np.array(all_labels)

    # Tính Rank-1 accuracy (simple leave-one-out)
    n = len(embeds)
    correct_rank1 = 0
    for i in range(n):
        query = embeds[i]
        sims  = embeds @ query
        sims[i] = -1  # exclude self
        best_idx = sims.argmax()
        if labels[best_idx] == labels[i]:
            correct_rank1 += 1

    rank1 = 100.0 * correct_rank1 / n
    log.info(f"Rank-1 Accuracy: {rank1:.2f}% ({correct_rank1}/{n})")
    return rank1

# ── Stage 2: Fine-tune on warehouse ──────────────────────────────────────────
def run_stage2():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # Load dataset
    dataset = WarehouseDataset(WAREHOUSE_ROOT, transform=TRAIN_TRANSFORM)
    val_dataset = WarehouseDataset(WAREHOUSE_ROOT, transform=VAL_TRANSFORM)

    # Split 80/20
    n_train = int(0.8 * len(dataset))
    n_val   = len(dataset) - n_train
    train_set, val_set = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    # Use val_dataset transform for val_set
    val_set.dataset.transform = VAL_TRANSFORM

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=32, shuffle=False,
                              num_workers=2, pin_memory=True)

    # Build model
    model = build_model(
        num_classes=dataset.num_classes,
        pretrained_path=STAGE1_WEIGHTS if Path(STAGE1_WEIGHTS).exists() else None
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-4, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=40, eta_min=1e-6
    )
    scaler = GradScaler()

    Path(MODEL_DIR).mkdir(exist_ok=True)
    best_rank1 = 0.0

    log.info("=" * 50)
    log.info(f"Stage 2: Fine-tune on {dataset.num_classes} warehouse classes")
    log.info(f"Train: {n_train} | Val: {n_val}")
    log.info("=" * 50)

    for epoch in range(1, 41):
        train_loss, train_acc = train(model, train_loader, optimizer, scaler, device, epoch)
        scheduler.step()

        if epoch % 5 == 0:
            rank1 = evaluate(model, val_loader, device)
            log.info(f"Epoch {epoch}: Loss={train_loss:.4f} "
                     f"TrainAcc={train_acc:.1f}% Rank-1={rank1:.2f}%")

            if rank1 > best_rank1:
                best_rank1 = rank1
                torch.save(model.state_dict(), STAGE2_WEIGHTS)
                log.info(f"  Saved best model: {STAGE2_WEIGHTS}")

    log.info(f"Training done. Best Rank-1: {best_rank1:.2f}%")
    log.info(f"Model saved: {STAGE2_WEIGHTS}")

# ── Stage 1: Pre-train trên Market-1501 ──────────────────────────────────────
def run_stage1():
    """
    Stage 1 dùng torchreid built-in training pipeline.
    Cần download Market-1501 trước:
    https://zheng-lab.cec.nus.edu.sg/Project/project_reid.html
    """
    log.info("Stage 1: Training on Market-1501")
    log.info(f"Market-1501 root: {MARKET1501_ROOT}")

    if not Path(MARKET1501_ROOT).exists():
        log.error(f"Market-1501 not found at {MARKET1501_ROOT}")
        log.error("Download: https://zheng-lab.cec.nus.edu.sg/Project/project_reid.html")
        log.error("Extract to: data/market1501/")
        return

    datamanager = torchreid.data.ImageDataManager(
        root=str(Path(MARKET1501_ROOT).parent),
        sources="market1501",
        targets="market1501",
        height=256,
        width=128,
        batch_size_train=32,
        batch_size_test=100,
        transforms=["random_flip", "random_erase"],
    )

    model = torchreid.models.build_model(
        name="osnet_x0_5",
        num_classes=datamanager.num_train_pids,
        pretrained=True,
    )
    model = model.cuda()

    optimizer = torchreid.optim.build_optimizer(
        model, optim="adam", lr=3.5e-4
    )
    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer, lr_scheduler="cosine", max_epoch=60
    )

    engine = torchreid.engine.ImageSoftmaxEngine(
        datamanager,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        label_smooth=True,
    )

    Path(MODEL_DIR).mkdir(exist_ok=True)
    engine.run(
        save_dir=MODEL_DIR,
        max_epoch=60,
        eval_freq=10,
        print_freq=50,
        test_only=False,
    )

    # Copy best model
    import shutil
    best = Path(MODEL_DIR) / "model" / "model.pth.tar-60"
    if best.exists():
        shutil.copy(best, STAGE1_WEIGHTS)
        log.info(f"Stage 1 model saved: {STAGE1_WEIGHTS}")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1", action="store_true",
                        help="Pre-train on Market-1501")
    parser.add_argument("--stage2", action="store_true",
                        help="Fine-tune on warehouse dataset")
    parser.add_argument("--eval",   action="store_true",
                        help="Evaluate current model")
    args = parser.parse_args()

    if args.stage1:
        run_stage1()
    elif args.stage2:
        run_stage2()
    elif args.eval:
        from config.settings import REID_MODEL_NAME
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataset = WarehouseDataset(WAREHOUSE_ROOT, transform=VAL_TRANSFORM)
        loader  = DataLoader(dataset, batch_size=32, shuffle=False)
        weights = STAGE2_WEIGHTS if Path(STAGE2_WEIGHTS).exists() else None
        model   = build_model(dataset.num_classes, weights).to(device)
        evaluate(model, loader, device)
    else:
        parser.print_help()
