#!/usr/bin/env python3
"""Train the development BEV segmenter and export one CPU/GPU ONNX model."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("artifacts/bev-dataset"))
    parser.add_argument("--output", type=Path, default=Path("models/bev_segmentation.onnx"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def require_training_runtime():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as error:
        raise SystemExit("PyTorch is not installed; install requirements-training.txt") from error
    return torch, nn, DataLoader, TensorDataset


def load_split(root: Path, manifest: dict, split: str) -> tuple[np.ndarray, np.ndarray]:
    features, labels = [], []
    for shard in manifest["shards"]:
        if shard["split"] != split:
            continue
        with np.load(root / shard["path"]) as data:
            features.append(np.asarray(data["bev"], dtype=np.float32))
            labels.append(np.asarray(data["mask"], dtype=np.float32)[:, None])
    if not features:
        raise SystemExit(f"dataset has no {split} shards")
    return np.concatenate(features), np.concatenate(labels)


def main() -> int:
    args = parse_args()
    if min(args.epochs, args.batch_size) < 1 or args.learning_rate <= 0:
        raise SystemExit("epochs, batch-size and learning-rate must be positive")
    torch, nn, DataLoader, TensorDataset = require_training_runtime()
    from ml.bev_model import BevSegmenter

    manifest_path = args.dataset / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_x, train_y = load_split(args.dataset, manifest, "train")
    validation_x, validation_y = load_split(args.dataset, manifest, "validation")
    if not np.any(validation_y):
        print("WARNING: validation split has no real positive run; metrics are pipeline diagnostics only")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    use_cuda = torch.cuda.is_available() if args.device == "auto" else args.device == "cuda"
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    device = torch.device("cuda" if use_cuda else "cpu")
    model = BevSegmenter(input_channels=train_x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    positives = float(train_y.sum())
    negatives = float(train_y.size - positives)
    positive_weight = torch.tensor([min(100.0, negatives / max(1.0, positives))], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=args.batch_size, shuffle=True,
    )
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model.logits(features), labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            prediction = model(torch.from_numpy(validation_x).to(device)).cpu().numpy()
        binary = prediction >= 0.5
        truth = validation_y >= 0.5
        intersection = int(np.count_nonzero(binary & truth))
        union = int(np.count_nonzero(binary | truth))
        false_positive_cells = int(np.count_nonzero(binary & ~truth))
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "validation_iou": intersection / union if union else None,
            "validation_false_positive_cells": false_positive_cells,
            "validation_false_positive_rate": false_positive_cells / binary.size,
        }
        history.append(row)
        print(json.dumps(row))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model = model.cpu().eval()
    example = torch.zeros((1, train_x.shape[1], train_x.shape[2], train_x.shape[3]), dtype=torch.float32)
    torch.onnx.export(
        model, example, args.output, input_names=["bev"], output_names=["probability"],
        opset_version=18, dynamic_axes={"bev": {0: "batch"}, "probability": {0: "batch"}},
        dynamo=False,
    )
    import onnx
    import onnxruntime as ort

    onnx.checker.check_model(onnx.load(str(args.output)))
    with torch.no_grad():
        reference = model(example).numpy()
    session = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    exported = session.run(None, {session.get_inputs()[0].name: example.numpy()})[0]
    maximum_export_error = float(np.max(np.abs(reference - exported)))
    if maximum_export_error > 1e-4:
        raise SystemExit(f"ONNX parity check failed: max error {maximum_export_error}")
    checksum = hashlib.sha256(args.output.read_bytes()).hexdigest()
    metadata = {
        "format": "metro-lidar-bev-model-v1",
        "sha256": checksum,
        "input_shape": [None, *train_x.shape[1:]],
        "output": "probability",
        "training_device": str(device),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "onnx_opset": 18,
        "maximum_export_error": maximum_export_error,
        "development_only": True,
        "reason": "no independent positive run is available",
        "history": history,
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"model": str(args.output), "sha256": checksum, "device": str(device)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
