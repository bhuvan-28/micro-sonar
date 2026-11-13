"""
scripts/train_model.py
=======================
Training script for the Micro-Sonar multi-task CNN.

Loads a pre-generated dataset (.npz), trains SonarNet, validates,
saves checkpoints, and exports the best model to ONNX + TFLite.

Usage:
    python scripts/train_model.py --dataset data/dataset.npz
    python scripts/train_model.py --dataset data/dataset.npz --backbone custom_cnn --epochs 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset, random_split
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[ERROR] PyTorch not installed. pip install torch")

from models.sonar_net import SonarLoss, build_model


def load_dataset(npz_path: str):
    data = np.load(npz_path)
    X = torch.from_numpy(data["features"])           # (N, 1, 128, 128)
    Y_dims = torch.from_numpy(data["dimensions"])    # (N, 3)
    Y_mat = torch.from_numpy(data["material_labels"])  # (N,)
    print(f"Loaded dataset: {X.shape[0]} samples from {npz_path}")
    return X, Y_dims, Y_mat


def train(
    dataset_path: str,
    backbone: str = "custom_cnn",
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    val_split: float = 0.15,
    test_split: float = 0.10,
    patience: int = 15,
    checkpoint_dir: str = "checkpoints",
    dim_weight: float = 1.0,
    mat_weight: float = 0.5,
) -> None:
    if not TORCH_AVAILABLE:
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    X, Y_dims, Y_mat = load_dataset(dataset_path)
    dataset = TensorDataset(X, Y_dims, Y_mat)

    n = len(dataset)
    n_test = int(n * test_split)
    n_val = int(n * val_split)
    n_train = n - n_test - n_val
    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"Split: train={n_train}, val={n_val}, test={n_test}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = build_model(backbone=backbone, pretrained=(backbone != "custom_cnn")).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {backbone} | Parameters: {n_params:,}")

    # Compute class weights from training split to handle any imbalance
    train_indices = train_ds.indices
    train_mat_labels = Y_mat[train_indices].numpy()
    class_counts = np.bincount(train_mat_labels, minlength=5).astype(np.float32)
    class_weights = torch.tensor(class_counts.sum() / (5 * class_counts + 1e-6)).to(device)
    print(f"Class weights: {class_weights.cpu().numpy().round(3)}")

    criterion = SonarLoss(dim_weight=dim_weight, mat_weight=mat_weight, class_weights=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        train_losses = []
        for xb, dims_b, mat_b in train_loader:
            xb = xb.to(device)
            dims_b = dims_b.to(device)
            mat_b = mat_b.to(device)
            optimizer.zero_grad()
            pred_dims, pred_mat = model(xb)
            loss, dim_loss, mat_loss = criterion(pred_dims, pred_mat, dims_b, mat_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
        scheduler.step()

        # --- Validate ---
        model.eval()
        val_losses, dim_maes = [], []
        correct_mat = 0
        total_mat = 0
        with torch.no_grad():
            for xb, dims_b, mat_b in val_loader:
                xb = xb.to(device)
                dims_b = dims_b.to(device)
                mat_b = mat_b.to(device)
                pred_dims, pred_mat = model(xb)
                loss, _, _ = criterion(pred_dims, pred_mat, dims_b, mat_b)
                val_losses.append(loss.item())
                dim_mae = torch.abs(pred_dims - dims_b).mean().item()
                dim_maes.append(dim_mae)
                correct_mat += (pred_mat.argmax(1) == mat_b).sum().item()
                total_mat += len(mat_b)

        avg_train = np.mean(train_losses)
        avg_val = np.mean(val_losses)
        avg_dim_mae = np.mean(dim_maes)
        mat_acc = correct_mat / total_mat * 100

        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train: {avg_train:.4f} | "
            f"Val: {avg_val:.4f} | "
            f"DimMAE: {avg_dim_mae:.3f}m | "
            f"MatAcc: {mat_acc:.1f}%"
        )

        # --- Checkpoint ---
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            stale_epochs = 0
            ckpt_path = Path(checkpoint_dir) / "best_sonar_model.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"  → Saved best checkpoint: {ckpt_path}")
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"\nEarly stopping triggered after {epoch} epochs.")
                break

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Best checkpoint saved at: {Path(checkpoint_dir) / 'best_sonar_model.pt'}")
    print("\nTo export to ONNX/TFLite, run:")
    print("  python models/export_tflite.py --checkpoint checkpoints/best_sonar_model.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--backbone", type=str, default="custom_cnn")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    args = parser.parse_args()

    train(
        dataset_path=args.dataset,
        backbone=args.backbone,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        checkpoint_dir=args.checkpoint_dir,
    )

