"""
cnn.py
-------
Small CNN for Galaxy10 DECaLS morphology classification.

Images are downsized 256x256 -> 128x128 for speed (galaxy morphology is
still very legible at that resolution). Standard train-time augmentation:
random flips + random rotation, since galaxy orientation on the sky is
arbitrary and the label should be rotation/flip invariant.

Usage:
    python cnn.py --data-dir ../data --epochs 25 --batch-size 64
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from data import load_raw, GALAXY10_CLASSES

IMG_SIZE = 128


class GalaxyDataset(Dataset):
    def __init__(self, images, labels, indices, train=False):
        self.images = images
        self.labels = labels
        self.indices = indices
        self.train = train

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        img = self.images[idx].astype(np.float32) / 255.0  # HWC, 256x256x3
        img = torch.from_numpy(img).permute(2, 0, 1)  # CHW
        img = F.interpolate(img.unsqueeze(0), size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False).squeeze(0)

        if self.train:
            if torch.rand(1).item() < 0.5:
                img = torch.flip(img, dims=[2])
            if torch.rand(1).item() < 0.5:
                img = torch.flip(img, dims=[1])
            k = int(torch.randint(0, 4, (1,)).item())
            img = torch.rot90(img, k, dims=[1, 2])

        # normalize to roughly zero mean / unit variance
        img = (img - 0.5) / 0.25
        label = int(self.labels[idx])
        return img, label, idx


class SmallCNN(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
        self.features = nn.Sequential(
            block(3, 32),    # 128 -> 64
            block(32, 64),   # 64 -> 32
            block(64, 128),  # 32 -> 16
            block(128, 256), # 16 -> 8
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.head(x)


def run_epoch(model, loader, device, optimizer=None):
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss, all_true, all_pred = 0.0, [], []
    for imgs, labels, _ in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.set_grad_enabled(train_mode):
            logits = model(imgs)
            loss = F.cross_entropy(logits, labels)
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        all_true.extend(labels.cpu().numpy())
        all_pred.extend(logits.argmax(1).cpu().numpy())
    acc = accuracy_score(all_true, all_pred)
    return total_loss / len(loader.dataset), acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="../data")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=6)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    images, labels = load_raw(args.data_dir)
    split = np.load(os.path.join(args.data_dir, "split_indices.npz"))
    train_idx, val_idx, test_idx = split["train_idx"], split["val_idx"], split["test_idx"]

    train_ds = GalaxyDataset(images, labels, train_idx, train=True)
    val_ds = GalaxyDataset(images, labels, val_idx, train=False)
    test_ds = GalaxyDataset(images, labels, test_idx, train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = SmallCNN(n_classes=10).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_val_acc, best_state, patience_ctr = 0.0, None, 0
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, device, optimizer=None)
        scheduler.step(val_acc)
        print(f"Epoch {epoch:3d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | val loss {val_loss:.4f} acc {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print("Early stopping.")
                break

    model.load_state_dict(best_state)
    model.to(device)

    os.makedirs(os.path.join(args.data_dir, "..", "models"), exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.data_dir, "..", "models", "cnn_best.pt"))

    # Final test evaluation, keeping per-sample predictions + original index
    # (needed by analysis.py to join back to hand-crafted features).
    model.eval()
    all_true, all_pred, all_idx, all_probs = [], [], [], []
    with torch.no_grad():
        for imgs, lbls, idxs in test_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            all_true.extend(lbls.numpy())
            all_pred.extend(logits.argmax(1).cpu().numpy())
            all_idx.extend(idxs.numpy())
            all_probs.append(probs)
    all_probs = np.concatenate(all_probs, axis=0)

    test_acc = accuracy_score(all_true, all_pred)
    test_bal_acc = balanced_accuracy_score(all_true, all_pred)
    print(f"\nFinal CNN test accuracy: {test_acc:.4f}  balanced_acc: {test_bal_acc:.4f}")

    np.savez(
        os.path.join(args.data_dir, "cnn_results.npz"),
        test_true=np.array(all_true),
        test_pred=np.array(all_pred),
        test_idx=np.array(all_idx),
        test_probs=all_probs,
        test_acc=test_acc,
        test_bal_acc=test_bal_acc,
    )
    print(f"Saved CNN predictions -> {os.path.join(args.data_dir, 'cnn_results.npz')}")


if __name__ == "__main__":
    main()
