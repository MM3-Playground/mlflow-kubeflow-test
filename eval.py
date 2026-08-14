import os
import cv2
import sys
import argparse
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from tqdm import tqdm
from matplotlib import pyplot as plt
import csv
import random
import json
from pathlib import Path
import pandas as pd

import torch
from torch import nn

import albumentations as A
from albumentations.pytorch import ToTensorV2

import mlflow

sys.path.insert(0, '..')
from models.Xception import *
from models.CNNDCT import *
from models.A import *

from utils.pilresize import PILResize
from utils.FCRDCT import *
from utils.tsne import *
from pipeline.helpers import write_portable_manifest


def read_paths(iut_paths_file, undersampling, subset):
    distribution = dict()
    n_min = None

    with open(iut_paths_file, 'r') as f:
        lines = f.readlines()
        for l in lines:
            parts = l.rstrip().split('\t')
            iut_path = parts[0]
            label = int(parts[1])

            if subset and subset not in parts[0]:
                continue

            if label not in distribution:
                distribution[label] = [iut_path]
            else:
                distribution[label].append(iut_path)

    for label in distribution:
        if n_min is None or len(distribution[label]) < n_min:
            n_min = len(distribution[label])

    iut_paths_labels = []

    for label in distribution:
        ll = distribution[label]

        if undersampling == 'all':
            for i in ll:
                iut_paths_labels.append((i, label))
        elif undersampling == 'min':
            picked = random.sample(ll, n_min)

            for p in picked:
                iut_paths_labels.append((p, label))
        else:
            print('Unsupported undersampling method {}!'.format(undersampling))
            sys.exit()

    return iut_paths_labels


def save_cm(y_true, y_pred, save_path):
    plt.figure()
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation')

    parser.add_argument("--id", type=str, help="run id")
    parser.add_argument("--iut_paths_file", type=str, default="/dataset/iut_files.txt")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--subset", type=str)
    parser.add_argument("--undersampling", type=str, default='all', choices=['all', 'min'])
    parser.add_argument('--out_dir', type=str, default='out')
    parser.add_argument('--model', default='xception', choices=['xception', 'cnndct', 'cnnpixel', 'ours'])
    parser.add_argument('--load_path', type=str, default="checkpoints/model.pth")

    parser.add_argument("--repo", type=str)
    parser.add_argument("--commit", type=str)
    parser.add_argument("--name", type=str)
    parser.add_argument("--dataset_root", type=str)

    parser.add_argument("--workspace", type=str)
    parser.add_argument("--experiment", type=str)

    return parser.parse_args()


def _load_model_weights(path: str, model: nn.Module, device: torch.device) -> None:
    """Load either a plain state_dict checkpoint or MLflow's serialized PyTorch model.

    The MLflow model artifact is produced by this project's own training run, so
    the full-model fallback is only used for that trusted artifact.
    """
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except Exception as exc:
        print(
            "[eval] weights_only=True could not load the MLflow model package; "
            "falling back to trusted full-model deserialization",
            flush=True,
        )
        payload = torch.load(path, map_location=device, weights_only=False)

    if isinstance(payload, nn.Module):
        model.load_state_dict(payload.state_dict())
    elif isinstance(payload, dict):
        model.load_state_dict(payload)
    else:
        raise TypeError(
            f"Unsupported checkpoint type from {path}: {type(payload).__name__}"
        )


def main(args: argparse.Namespace | None = None) -> dict:
    if args is None:
        args = parse_args()

    mlflow.set_workspace(args.workspace)
    mlflow.set_experiment(args.experiment)

    mlflow_run = mlflow.start_run(run_name=f"eval-{args.id}")
    parent_run_id = os.environ.get("PARENT_MLFLOW_RUN_ID")
    if parent_run_id:
        mlflow.set_tag("lineage.parent_run_id", parent_run_id)

    mlflow.log_params({
        "run_id": args.id,
        "image_size": args.image_size,
        "iut_paths_file": args.iut_paths_file,
        "subset": args.subset,
        "undersampling": args.undersampling,
        "model": args.model,
        "load_path": args.load_path,
    })

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model == 'xception':
        model = Xception().to(device)
    elif args.model in ('cnndct', 'cnnpixel'):
        model = CNNDCT(args.image_size).to(device)
    elif args.model == 'ours':
        model = Attributor(args.image_size).to(device)
    else:
        print("Unrecognized model %s" % args.model)
        sys.exit()

    if args.load_path is not None and os.path.exists(args.load_path):
        print('Load pretrained model: {}'.format(args.load_path), flush=True)
        _load_model_weights(args.load_path, model, device)
    else:
        print("%s not exist" % args.load_path)
        sys.exit()

    model.eval()

    if not os.path.exists(args.iut_paths_file):
        print("%s not exists, quit" % args.iut_paths_file)
        sys.exit()

    if args.subset:
        print("Evaluation on subset {}".format(args.subset))

    iut_paths_labels = read_paths(args.iut_paths_file, args.undersampling, args.subset)
    print("Eval set size is {}!".format(len(iut_paths_labels)), flush=True)

    mlflow.log_artifact(args.iut_paths_file, "datasets")
    if args.dataset_root:
        portable_test = write_portable_manifest(
            args.iut_paths_file,
            Path(args.out_dir) / "test_datalad.txt",
            args.dataset_root,
        )
        mlflow.log_artifact(str(portable_test), "datasets/portable")

    dataframe = pd.DataFrame({
        "input_image_paths": [iut_path for iut_path, _ in iut_paths_labels],
        "labels": [label for _, label in iut_paths_labels]
    })
    dataset = mlflow.data.from_pandas(
        dataframe,
        source=args.repo,
        digest=args.commit[:36],
        name=args.name + "-test"
    )
    mlflow.log_input(dataset, "test")

    mlflow.log_params({
        "repo": args.repo,
        "commit": args.commit,
        "name": args.name,
    })

    print("Predicted maps will be saved in :%s" % args.out_dir, flush=True)
    os.makedirs(args.out_dir, exist_ok=True)
    if args.subset is None:
        os.makedirs(os.path.join(args.out_dir, 'images'), exist_ok=True)

    if args.undersampling == 'min':
        save_path = os.path.join(args.out_dir, 'paths_file_eval.txt')
        with open(save_path, 'w') as f:
            for (iut_path, label) in iut_paths_labels:
                f.write(iut_path + '\t' + str(label) + '\n')
        print('Eval paths file saved to %s' % save_path, flush=True)
        mlflow.log_artifact(save_path, "datasets")

    if args.subset is None:
        f_csv = open(os.path.join(args.out_dir, 'pred.csv'), 'w', newline='')
        writer = csv.writer(f_csv)
        writer.writerow(['Image', 'Pred', 'True', 'Correct'])

    if args.model in ('xception', 'cnnpixel'):
        transform = A.Compose([
            A.Normalize(mean=0.0, std=1.0),
            ToTensorV2()
        ])
    else:
        transform = A.Compose([
            A.Normalize(mean=0.0, std=1.0),
            ToTensorV2(),
            DCT(p=1.0, log=True, factor=1)
        ])

    y_pred = []
    y_true = []

    for iut_path, lab in tqdm(iut_paths_labels, mininterval=60):
        try:
            img = cv2.cvtColor(cv2.imread(iut_path), cv2.COLOR_BGR2RGB)
        except Exception:
            print('Failed to load image {}'.format(iut_path))
            continue
        if img is None:
            print('Failed to load image {}'.format(iut_path))
            continue

        img = transform(image=img)['image'].to(device)

        with torch.no_grad():
            out = model(img.unsqueeze(0))
        y = int(torch.sigmoid(out).item() > 0.5)

        y_pred.append(y)
        y_true.append(lab)

        if args.subset is None:
            writer.writerow([iut_path, y, lab, y == lab])

    accuracy = accuracy_score(y_true, y_pred)
    print("acc%s: %.4f" % ((' (' + args.subset + ')' if args.subset else ''), accuracy), flush=True)

    save_path = os.path.join(
        args.out_dir,
        'cm' + ('_' + args.subset if args.subset else '') + '.png'
    )
    save_cm(y_true, y_pred, save_path)

    mlflow.log_metric("accuracy", float(accuracy))
    mlflow.log_artifacts(args.out_dir, "results")

    if args.subset is None:
        f_csv.close()

    result_dir = Path(os.environ.get("SAVE_DIR", args.out_dir)) / "pipeline-results"
    result_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "execution_id": str(args.id),
        "mlflow_run_id": mlflow_run.info.run_id,
        "parent_mlflow_run_id": parent_run_id,
        "accuracy": float(accuracy),
        "output_dir": str(Path(args.out_dir).resolve()),
    }
    result_path = result_dir / f"eval-{args.id}.json"
    result_path.write_text(json.dumps(result, indent=2))
    mlflow.log_artifact(str(result_path), "pipeline")
    mlflow.end_run()
    return result


if __name__ == '__main__':
    main()