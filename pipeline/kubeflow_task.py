#!/usr/bin/env python3
"""Thin Kubeflow orchestration adapter.

The cluster admission policy prepares /workspace/code and /workspace/venv.
This module handles only orchestration-specific preparation and then calls the
existing MLflow-aware training/evaluation Python functions directly.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import subprocess
import zipfile
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlretrieve

import boto3
import mlflow

import eval as eval_module
import train_local


CODE_DIR = Path("/workspace/code")
DATA_DIR = Path("/tmp/data")
MANIFEST_DIR = Path("/tmp/manifests")
MANIFEST_BUNDLE = Path("/tmp/manifest-bundle.zip")
TRAIN_SAVE_DIR = Path("/tmp/work/run")
EVAL_OUT_DIR = Path("/tmp/eval")
MODEL_DOWNLOAD_DIR = Path("/tmp/load-model")


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    """Run infrastructure CLIs such as git/DataLad.

    Training and evaluation themselves are called directly as Python functions.
    """
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run([str(x) for x in cmd], cwd=cwd, check=True)


def write_output(path: str, value: object) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(str(value), encoding="utf-8")


def git_askpass_env() -> None:
    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        return

    script = Path("/tmp/git-askpass.sh")
    script.write_text(
        """#!/bin/sh
case "$1" in
  *sername*) printf '%s\\n' "${GITHUB_USERNAME:-x-access-token}" ;;
  *) printf '%s\\n' "$GITHUB_TOKEN" ;;
esac
""",
        encoding="utf-8",
    )
    script.chmod(0o700)
    os.environ["GIT_ASKPASS"] = str(script)
    os.environ["GIT_TERMINAL_PROMPT"] = "0"


def download_manifest_bundle(uri: str) -> Path:
    """Download the already-created manifest bundle without using dsl.importer."""
    MANIFEST_BUNDLE.unlink(missing_ok=True)

    if uri.startswith("s3://"):
        parsed = urlparse(uri)
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        if not bucket or not key:
            raise ValueError(f"Invalid S3 manifest URI: {uri}")

        endpoint = os.environ.get("AWS_ENDPOINT_URL")
        s3 = boto3.client(
            "s3",
            endpoint_url=endpoint,
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
        )
        print(f"[manifest] Downloading {uri}", flush=True)
        s3.download_file(bucket, key, str(MANIFEST_BUNDLE))
        return MANIFEST_BUNDLE

    if uri.startswith(("http://", "https://")):
        print(f"[manifest] Downloading {uri}", flush=True)
        urlretrieve(uri, MANIFEST_BUNDLE)
        return MANIFEST_BUNDLE

    source = Path(uri.removeprefix("file://"))
    if not source.exists():
        raise FileNotFoundError(f"Manifest bundle does not exist: {uri}")
    shutil.copyfile(source, MANIFEST_BUNDLE)
    return MANIFEST_BUNDLE


def clone_dataset(repo: str, commit: str) -> str:
    shutil.rmtree(DATA_DIR, ignore_errors=True)
    run(["datalad", "clone", repo, str(DATA_DIR)])

    if commit:
        run(["git", "checkout", "--detach", commit], cwd=DATA_DIR)

    resolved = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=DATA_DIR, text=True
    ).strip()
    run(["datalad", "get", "-r", "."], cwd=DATA_DIR)
    return resolved


def extract_manifests(bundle: Path) -> None:
    shutil.rmtree(MANIFEST_DIR, ignore_errors=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(bundle, "r") as archive:
        archive.extractall(MANIFEST_DIR)

    # Keep current POC behavior. Relative paths are rooted at /tmp/data.
    # Absolute paths are intentionally left unchanged.
    for name in ("train_datalad.txt", "val_datalad.txt", "test_datalad.txt"):
        path = MANIFEST_DIR / name
        if not path.exists():
            continue

        rewritten: list[str] = []
        for raw in path.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            item, label = raw.split("\t", 1)
            candidate = Path(item)
            resolved = candidate if candidate.is_absolute() else DATA_DIR / candidate
            rewritten.append(f"{resolved.resolve()}\t{label}")

        path.write_text("\n".join(rewritten) + "\n", encoding="utf-8")


def download_model(workspace: str, model_uri: str) -> Path:
    shutil.rmtree(MODEL_DOWNLOAD_DIR, ignore_errors=True)
    MODEL_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    mlflow.set_workspace(workspace)
    downloaded = Path(
        mlflow.artifacts.download_artifacts(
            artifact_uri=model_uri,
            dst_path=str(MODEL_DOWNLOAD_DIR),
        )
    )
    checkpoints = list(downloaded.rglob("*.pth"))
    if not checkpoints:
        raise RuntimeError(f"No .pth checkpoint found in {downloaded}")
    return checkpoints[0]


def resolved_code_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=CODE_DIR, text=True
    ).strip()


def train(args: argparse.Namespace) -> None:
    git_askpass_env()

    bundle = download_manifest_bundle(args.manifest_bundle_uri)
    dataset_commit = clone_dataset(args.dataset_repo_url, args.dataset_commit)
    extract_manifests(bundle)

    train_manifest = MANIFEST_DIR / "train_datalad.txt"
    test_manifest = MANIFEST_DIR / "test_datalad.txt"
    val_manifest = MANIFEST_DIR / "val_datalad.txt"

    if not train_manifest.exists() or not test_manifest.exists():
        raise RuntimeError(
            "Manifest bundle must contain train_datalad.txt and test_datalad.txt"
        )

    shutil.rmtree(TRAIN_SAVE_DIR, ignore_errors=True)
    TRAIN_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    code_commit = resolved_code_commit()
    dataset_name = DATA_DIR.name
    from datetime import datetime, timezone
    execution_id = "train-" + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")

    os.environ["CODE_REPO"] = args.code_repo_url
    os.environ["CODE_COMMIT"] = code_commit
    os.environ["PIPELINE_KIND"] = args.pipeline_kind

    settings = {
        "run_name": args.run_name,
        "seed": args.seed,
        "model": args.model,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "optimizer": args.optimizer,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "factor": args.factor,
        "patience": args.patience,
        "early_stopping_patience": args.early_stopping_patience,
        "n_c_samples": args.n_c_samples,
        "val_n_c_samples": args.val_n_c_samples,
        "pipeline_kind": args.pipeline_kind,
        "mlflow_workspace": args.mlflow_workspace,
        "mlflow_experiment": args.mlflow_experiment,
    }
    os.environ["PIPELINE_SETTINGS_B64"] = base64.b64encode(
        json.dumps(settings, separators=(",", ":")).encode()
    ).decode()

    load_path = None
    if args.load_model_uri:
        load_path = str(download_model(args.mlflow_workspace, args.load_model_uri))

    train_args = argparse.Namespace(
        id=execution_id,
        run_name=args.run_name,
        seed=args.seed,
        paths_file=str(train_manifest),
        val_paths_file=str(val_manifest) if val_manifest.exists() else None,
        test_paths_file=str(test_manifest),
        n_c_samples=args.n_c_samples if args.n_c_samples >= 0 else None,
        val_n_c_samples=args.val_n_c_samples if args.val_n_c_samples >= 0 else None,
        image_size=args.image_size,
        workers=args.workers,
        batch_size=args.batch_size,
        model=args.model,
        load_path=load_path,
        optim=args.optimizer,
        factor=args.factor,
        patience=args.patience,
        lr=args.learning_rate,
        n_epochs=args.epochs,
        n_early=args.early_stopping_patience,
        save_dir=str(TRAIN_SAVE_DIR),
        repo=args.dataset_repo_url,
        commit=dataset_commit,
        name=dataset_name,
        dataset_root=str(DATA_DIR),
        workspace=args.mlflow_workspace,
        experiment=args.mlflow_experiment,
        device="cpu",
    )

    # Direct Python call: no subprocess for the training entry point.
    result = train_local.main(train_args)

    write_output(args.out_mlflow_run_id, result["mlflow_run_id"])
    write_output(args.out_model_uri, result["model_uri"])
    write_output(args.out_code_commit, code_commit)
    write_output(args.out_dataset_commit, dataset_commit)
    write_output(args.out_dataset_name, dataset_name)


def evaluate(args: argparse.Namespace) -> None:
    git_askpass_env()

    bundle = download_manifest_bundle(args.manifest_bundle_uri)
    dataset_commit = clone_dataset(args.dataset_repo_url, args.dataset_commit)
    extract_manifests(bundle)

    test_manifest = MANIFEST_DIR / "test_datalad.txt"
    if not test_manifest.exists():
        raise RuntimeError("Manifest bundle must contain test_datalad.txt")

    checkpoint = download_model(args.mlflow_workspace, args.model_uri)

    shutil.rmtree(EVAL_OUT_DIR, ignore_errors=True)
    EVAL_OUT_DIR.mkdir(parents=True, exist_ok=True)

    from datetime import datetime, timezone
    execution_id = "eval-" + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")

    os.environ["PARENT_MLFLOW_RUN_ID"] = args.parent_mlflow_run_id
    os.environ["PIPELINE_KIND"] = args.pipeline_kind

    eval_args = argparse.Namespace(
        id=execution_id,
        iut_paths_file=str(test_manifest),
        image_size=args.image_size,
        subset=None,
        undersampling="all",
        out_dir=str(EVAL_OUT_DIR),
        model=args.model,
        load_path=str(checkpoint),
        repo=args.dataset_repo_url,
        commit=dataset_commit,
        name=args.dataset_name,
        dataset_root=str(DATA_DIR),
        workspace=args.mlflow_workspace,
        experiment=args.mlflow_experiment,
    )

    # Direct Python call: no subprocess for the evaluation entry point.
    result = eval_module.main(eval_args)

    write_output(args.out_accuracy, float(result["accuracy"]))
    write_output(args.out_mlflow_run_id, result["mlflow_run_id"])


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--code-repo-url", required=True)
    common.add_argument("--code-commit", default="")
    common.add_argument("--dataset-repo-url", required=True)
    common.add_argument("--dataset-commit", default="")
    common.add_argument("--manifest-bundle-uri", required=True)
    common.add_argument("--pipeline-kind", required=True)
    common.add_argument("--model", required=True)
    common.add_argument("--image-size", type=int, required=True)
    common.add_argument("--mlflow-workspace", required=True)
    common.add_argument("--mlflow-experiment", required=True)

    t = sub.add_parser("train", parents=[common])
    t.add_argument("--run-name", required=True)
    t.add_argument("--seed", type=int, required=True)
    t.add_argument("--batch-size", type=int, required=True)
    t.add_argument("--workers", type=int, required=True)
    t.add_argument("--optimizer", required=True)
    t.add_argument("--learning-rate", type=float, required=True)
    t.add_argument("--epochs", type=int, required=True)
    t.add_argument("--factor", type=float, required=True)
    t.add_argument("--patience", type=int, required=True)
    t.add_argument("--early-stopping-patience", type=int, required=True)
    t.add_argument("--n-c-samples", type=int, required=True)
    t.add_argument("--val-n-c-samples", type=int, required=True)
    t.add_argument("--load-model-uri", default="")
    t.add_argument("--out-mlflow-run-id", required=True)
    t.add_argument("--out-model-uri", required=True)
    t.add_argument("--out-code-commit", required=True)
    t.add_argument("--out-dataset-commit", required=True)
    t.add_argument("--out-dataset-name", required=True)
    t.set_defaults(func=train)

    e = sub.add_parser("evaluate", parents=[common])
    e.add_argument("--dataset-name", required=True)
    e.add_argument("--parent-mlflow-run-id", required=True)
    e.add_argument("--model-uri", required=True)
    e.add_argument("--out-accuracy", required=True)
    e.add_argument("--out-mlflow-run-id", required=True)
    e.set_defaults(func=evaluate)

    return p


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
