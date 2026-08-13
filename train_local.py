from __future__ import annotations

import argparse
import base64
import json
import os
import random
from pathlib import Path

import mlflow
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets.dataset import AnimeDataset
from models.A import Attributor
from models.CNNDCT import CNNDCT
from models.Xception import Xception
from pipeline.helpers import write_portable_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-process local CPU/GPU training")
    parser.add_argument("--id", required=True)
    parser.add_argument("--run_name", default="freq")
    parser.add_argument("--seed", type=int, default=3721)
    parser.add_argument("--paths_file", required=True)
    parser.add_argument("--val_paths_file")
    parser.add_argument("--test_paths_file", required=True)
    parser.add_argument("--n_c_samples", type=int)
    parser.add_argument("--val_n_c_samples", type=int)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--model", default="ours", choices=["xception", "cnndct", "cnnpixel", "ours"])
    parser.add_argument("--load_path")
    parser.add_argument("--optim", choices=["adam", "adamw"], default="adamw")
    parser.add_argument("--factor", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n_epochs", type=int, default=2)
    parser.add_argument("--n_early", type=int, default=10)
    parser.add_argument("--save_dir", default=".")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    return parser.parse_args()


def build_model(name: str, image_size: int, device: torch.device) -> nn.Module:
    if name == "xception":
        model = Xception()
    elif name in {"cnndct", "cnnpixel"}:
        model = CNNDCT(image_size)
    elif name == "ours":
        model = Attributor(image_size)
    else:
        raise ValueError(f"Unsupported model: {name}")
    return model.to(device)


def make_loader(args: argparse.Namespace, *, validation: bool) -> DataLoader | None:
    paths_file = args.val_paths_file if validation else args.paths_file
    if validation and not paths_file:
        return None
    dct = args.model in {"cnndct", "ours"}
    dataset = AnimeDataset(
        0,
        paths_file,
        args.image_size,
        args.id,
        dct,
        args.val_n_c_samples if validation else args.n_c_samples,
        validation,
        args.repo,
        args.commit,
        args.name,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=not validation,
        num_workers=args.workers,
        pin_memory=False,
        drop_last=False,
    )


def main(args: argparse.Namespace | None = None) -> dict:
    if args is None:
        args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    save_dir = Path(args.save_dir).resolve()
    checkpoint_dir = save_dir / "checkpoints" / f"{args.id}_{args.run_name}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_workspace(args.workspace)
    mlflow.set_experiment(args.experiment)
    run = mlflow.start_run(run_name=f"train-{args.run_name}-{args.id}")

    try:
        settings_b64 = os.environ.get("PIPELINE_SETTINGS_B64", "")
        settings_json = base64.b64decode(settings_b64).decode() if settings_b64 else "{}"
        mlflow.set_tags(
            {
                "mlflow.source.git.repoURL": os.environ.get("CODE_REPO", ""),
                "mlflow.source.git.commit": os.environ.get("CODE_COMMIT", ""),
                "pipeline.settings_json": settings_json,
                "pipeline.kind": os.environ.get("PIPELINE_KIND", "new"),
                "pipeline.execution_backend": "kubeflow",
            }
        )
        mlflow.log_params(
            {
                "run_id": args.id,
                "run_name": args.run_name,
                "seed": args.seed,
                "model": args.model,
                "image_size": args.image_size,
                "batch_size": args.batch_size,
                "optimizer": args.optim,
                "learning_rate": args.lr,
                "epochs": args.n_epochs,
                "device": str(device),
                "train_paths_file": args.paths_file,
                "val_paths_file": args.val_paths_file or "",
                "load_path": args.load_path or "",
            }
        )
        mlflow.log_artifact(args.paths_file, "datasets")
        if args.val_paths_file:
            mlflow.log_artifact(args.val_paths_file, "datasets")

        model = build_model(args.model, args.image_size, device)
        if args.load_path:
            model.load_state_dict(torch.load(args.load_path, map_location=device))

        optimizer_cls = torch.optim.Adam if args.optim == "adam" else torch.optim.AdamW
        optimizer = optimizer_cls(model.parameters(), lr=args.lr)
        scheduler = None
        if args.val_paths_file and args.patience:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=args.factor, patience=args.patience
            )

        train_loader = make_loader(args, validation=False)
        val_loader = make_loader(args, validation=True)
        if not train_loader or len(train_loader.dataset) == 0:
            raise RuntimeError("The training dataset is empty")

        criterion = nn.BCEWithLogitsLoss()
        best_val_loss = float("inf")
        best_checkpoint: Path | None = None
        no_improvement = 0

        for epoch in range(args.n_epochs):
            model.train()
            train_loss_sum = 0.0
            train_batches = 0
            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.float().unsqueeze(1).to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss_sum += float(loss.item())
                train_batches += 1

            train_loss = train_loss_sum / max(train_batches, 1)
            metrics = {
                "train_loss": train_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }

            improved = val_loader is None
            if val_loader is not None:
                model.eval()
                val_loss_sum = 0.0
                val_batches = 0
                with torch.no_grad():
                    for images, labels in val_loader:
                        images = images.to(device)
                        labels = labels.float().unsqueeze(1).to(device)
                        outputs = model(images)
                        val_loss_sum += float(criterion(outputs, labels).item())
                        val_batches += 1
                val_loss = val_loss_sum / max(val_batches, 1)
                metrics["val_loss"] = val_loss
                improved = val_loss <= best_val_loss
                if improved:
                    best_val_loss = val_loss
                    no_improvement = 0
                else:
                    no_improvement += 1
                if scheduler is not None:
                    scheduler.step(val_loss)

            last_checkpoint = checkpoint_dir / f"{args.id}_last_{epoch}.pth"
            torch.save(model.state_dict(), last_checkpoint)
            if improved:
                best_checkpoint = checkpoint_dir / f"{args.id}_best_{epoch}.pth"
                torch.save(model.state_dict(), best_checkpoint)
                mlflow.log_artifact(str(best_checkpoint), "best_models")

            mlflow.log_metrics(metrics, step=epoch)
            print(f"Epoch {epoch}: {metrics}")
            if val_loader is not None and no_improvement > args.n_early:
                print("Early stopping")
                break

        selected_checkpoint = best_checkpoint or last_checkpoint
        model.load_state_dict(torch.load(selected_checkpoint, map_location=device))
        model_info = mlflow.pytorch.log_model(
            model,
            name="model",
            serialization_format="pickle",
            code_paths=[str(Path.cwd())],
        )

        portable_dir = save_dir / "portable-manifests" / args.id
        portable_train = write_portable_manifest(
            Path(args.paths_file).resolve().parent / f"cond_paths_file_{args.id}_train.txt",
            portable_dir / f"cond_paths_file_{args.id}_train.txt",
            args.dataset_root,
        )
        mlflow.log_artifact(str(portable_train), "datasets/portable")

        portable_val = None
        if args.val_paths_file:
            portable_val = write_portable_manifest(
                Path(args.val_paths_file).resolve().parent / f"cond_paths_file_{args.id}_val.txt",
                portable_dir / f"cond_paths_file_{args.id}_val.txt",
                args.dataset_root,
            )
            mlflow.log_artifact(str(portable_val), "datasets/portable")

        for source, name in [
            (args.paths_file, "train_datalad.txt"),
            (args.val_paths_file, "val_datalad.txt"),
            (args.test_paths_file, "test_datalad.txt"),
        ]:
            if source:
                portable_original = write_portable_manifest(
                    source, portable_dir / name, args.dataset_root
                )
                mlflow.log_artifact(str(portable_original), "datasets/portable")

        result_dir = save_dir / "pipeline-results"
        result_dir.mkdir(parents=True, exist_ok=True)
        result = {
            "execution_id": args.id,
            "mlflow_run_id": run.info.run_id,
            "best_checkpoint": str(selected_checkpoint.resolve()),
            "model_uri": model_info.model_uri,
            "train_conditioned_paths_file": str(
                Path(args.paths_file).resolve().parent
                / f"cond_paths_file_{args.id}_train.txt"
            ),
            "val_conditioned_paths_file": str(
                Path(args.val_paths_file).resolve().parent
                / f"cond_paths_file_{args.id}_val.txt"
            )
            if args.val_paths_file
            else None,
        }
        result_path = result_dir / f"train-{args.id}.json"
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(result_path), "pipeline")
        return result
    finally:
        mlflow.end_run()


if __name__ == "__main__":
    main()
