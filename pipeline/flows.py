from typing import NamedTuple
import os
from kfp import dsl, kubernetes
from kfp.dsl import Dataset, Input

def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"{name} must be exported before compiling the pipeline"
        )
    return value


RUNTIME_IMAGE = _required_env("KUBEFLOW_RUNTIME_IMAGE")
IMAGE_PULL_SECRET = _required_env("KUBEFLOW_IMAGE_PULL_SECRET")

MLFLOW_TRACKING_URI = _required_env("MLFLOW_TRACKING_URI")
MLFLOW_TRACKING_USERNAME = _required_env("MLFLOW_TRACKING_USERNAME")
MLFLOW_TRACKING_PASSWORD = _required_env("MLFLOW_TRACKING_PASSWORD")


@dsl.component(base_image="python:3.11-slim")
def write_manifest_bundle(
    train_b64: str,
    val_b64: str,
    test_b64: str,
    manifests: dsl.Output[Dataset],
):
    import base64
    import zipfile
    from pathlib import Path

    target = Path(manifests.path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, value in [
            ("train_datalad.txt", train_b64),
            ("val_datalad.txt", val_b64),
            ("test_datalad.txt", test_b64),
        ]:
            if value:
                archive.writestr(name, base64.b64decode(value))


@dsl.pipeline(name="upload-manifest-bundle")
def upload_manifest_bundle_pipeline(train_b64: str, val_b64: str, test_b64: str):
    write_manifest_bundle(train_b64=train_b64, val_b64=val_b64, test_b64=test_b64)


@dsl.component(base_image=RUNTIME_IMAGE)
def train_local(
    code_repo_url: str,
    dataset_repo_url: str,
    code_commit: str,
    dataset_commit: str,
    manifest_bundle: Input[Dataset],
    pipeline_kind: str,
    run_name: str,
    seed: int,
    model: str,
    image_size: int,
    batch_size: int,
    workers: int,
    optimizer: str,
    learning_rate: float,
    epochs: int,
    factor: float,
    patience: int,
    early_stopping_patience: int,
    n_c_samples: int,
    val_n_c_samples: int,
    load_model_uri: str,
    mlflow_workspace: str,
    mlflow_experiment: str,
) -> NamedTuple(
    "Outputs",
    [
        ("mlflow_run_id", str),
        ("model_uri", str),
        ("best_checkpoint", str),
        ("code_commit_resolved", str),
        ("dataset_commit_resolved", str),
        ("dataset_name", str),
    ],
):
    import base64
    import json
    import os
    import stat
    import subprocess
    import sys
    import tempfile
    import time
    import zipfile
    from pathlib import Path
    from typing import NamedTuple

    def git_auth_env():
        env = os.environ.copy()
        env["GIT_TERMINAL_PROMPT"] = "0"
        token = env.get("GITHUB_TOKEN", "")
        if not token:
            return env, None
        handle = tempfile.NamedTemporaryFile("w", delete=False, prefix="git-askpass-")
        handle.write(
            "#!/bin/sh\n"
            "case \"$1\" in\n"
            "  *sername*) printf '%s\\n' \"$GITHUB_USERNAME\" ;;\n"
            "  *)         printf '%s\\n' \"$GITHUB_TOKEN\" ;;\n"
            "esac\n"
        )
        handle.close()
        os.chmod(handle.name, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
        env["GIT_ASKPASS"] = handle.name
        return env, handle.name

    auth_env, askpass_path = git_auth_env()

    def cmd(command, cwd=None, use_git_auth=False, extra_env=None):
        env = auth_env.copy() if use_git_auth else os.environ.copy()
        if extra_env:
            env.update(extra_env)
        subprocess.run(command, cwd=cwd, env=env, check=True)

    def run(command, cwd=None, use_git_auth=False):
        return subprocess.check_output(
            command,
            cwd=cwd,
            env=auth_env if use_git_auth else None,
            text=True,
        ).strip()

    code = Path("/tmp/code")
    data = Path("/tmp/data")
    cmd(["git", "config", "--global", "user.name", "Kubeflow Pipeline"])
    cmd(["git", "config", "--global", "user.email", "kubeflow@localhost"])

    cmd(["git", "clone", code_repo_url, str(code)], use_git_auth=True)
    resolved_code_commit = code_commit or run(["git", "rev-parse", "HEAD"], code, True)
    cmd(["git", "checkout", "--detach", resolved_code_commit], code, True)
    resolved_code_commit = run(["git", "rev-parse", "HEAD"], code, True)

    cmd(["datalad", "clone", dataset_repo_url, str(data)], use_git_auth=True)
    resolved_dataset_commit = dataset_commit or run(["git", "rev-parse", "HEAD"], data, True)
    cmd(["git", "checkout", "--detach", resolved_dataset_commit], data, True)
    resolved_dataset_commit = run(["git", "rev-parse", "HEAD"], data, True)
    dataset_name = data.name
    cmd(["datalad", "get", "-r", "."], data, True)

    venv = Path("/tmp/venv")
    cmd([sys.executable, "-m", "venv", str(venv)])
    runtime_python = str(venv / "bin" / "python")
    cmd([runtime_python, "-m", "pip", "install", "-r", str(code / "requirements.txt")])

    bundle_dir = Path("/tmp/manifest-bundle")
    bundle_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(manifest_bundle.path, "r") as archive:
        archive.extractall(bundle_dir)

    material = Path("/tmp/manifests")
    material.mkdir(parents=True, exist_ok=True)
    for name in ["train_datalad.txt", "val_datalad.txt", "test_datalad.txt"]:
        src = bundle_dir / name
        if not src.exists():
            continue
        lines = []
        for raw in src.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            path, label = raw.split("\t")
            candidate = Path(path)
            resolved = candidate if candidate.is_absolute() else data / candidate
            lines.append(f"{resolved.resolve()}\t{label}")
        (material / name).write_text("\n".join(lines) + "\n", encoding="utf-8")

    execution_id = "train-" + time.strftime("%Y%m%d%H%M%S")
    save_dir = Path("/tmp/work/run")
    settings = {
        "run_name": run_name,
        "seed": seed,
        "model": model,
        "image_size": image_size,
        "batch_size": batch_size,
        "workers": workers,
        "optimizer": optimizer,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "factor": factor,
        "patience": patience,
        "early_stopping_patience": early_stopping_patience,
        "n_c_samples": n_c_samples,
        "val_n_c_samples": val_n_c_samples,
        "pipeline_kind": pipeline_kind,
        "mlflow_workspace": mlflow_workspace,
        "mlflow_experiment": mlflow_experiment,
    }
    settings_b64 = base64.b64encode(
        json.dumps(settings, separators=(",", ":")).encode()
    ).decode()

    command = [
        runtime_python, "-u", "train_local.py",
        "--id", execution_id,
        "--run_name", run_name,
        "--seed", str(seed),
        "--save_dir", str(save_dir),
        "--batch_size", str(batch_size),
        "--workers", str(workers),
        "--model", model,
        "--image_size", str(image_size),
        "--optim", optimizer,
        "--factor", str(factor),
        "--patience", str(patience),
        "--paths_file", str(material / "train_datalad.txt"),
        "--test_paths_file", str(material / "test_datalad.txt"),
        "--n_epochs", str(epochs),
        "--n_early", str(early_stopping_patience),
        "--lr", str(learning_rate),
        "--device", "cpu",
        "--repo", dataset_repo_url,
        "--commit", resolved_dataset_commit,
        "--name", dataset_name,
        "--dataset_root", str(data),
        "--workspace", mlflow_workspace,
        "--experiment", mlflow_experiment,
    ]
    if (material / "val_datalad.txt").exists():
        command += ["--val_paths_file", str(material / "val_datalad.txt")]
    if n_c_samples >= 0:
        command += ["--n_c_samples", str(n_c_samples)]
    if val_n_c_samples >= 0:
        command += ["--val_n_c_samples", str(val_n_c_samples)]
    if load_model_uri:
        command += ["--load_path", load_model_uri]

    cmd(
        command,
        code,
        extra_env={
            "CODE_REPO": code_repo_url,
            "CODE_COMMIT": resolved_code_commit,
            "PIPELINE_SETTINGS_B64": settings_b64,
            "PIPELINE_KIND": pipeline_kind,
        },
    )

    result = json.loads(
        (save_dir / "pipeline-results" / f"train-{execution_id}.json").read_text()
    )
    if askpass_path:
        Path(askpass_path).unlink(missing_ok=True)

    return (
        result["mlflow_run_id"],
        result["model_uri"],
        result["best_checkpoint"],
        resolved_code_commit,
        resolved_dataset_commit,
        dataset_name,
    )


@dsl.component(base_image=RUNTIME_IMAGE)
def evaluate_local(
    code_repo_url: str,
    dataset_repo_url: str,
    code_commit: str,
    dataset_commit: str,
    dataset_name: str,
    manifest_bundle: Input[Dataset],
    parent_mlflow_run_id: str,
    model_uri: str,
    pipeline_kind: str,
    model: str,
    image_size: int,
    mlflow_workspace: str,
    mlflow_experiment: str,
) -> NamedTuple("Outputs", [("accuracy", float), ("mlflow_run_id", str)]):
    import json
    import os
    import stat
    import subprocess
    import sys
    import tempfile
    import time
    import zipfile
    from pathlib import Path
    from typing import NamedTuple
    import mlflow

    def git_auth_env():
        env = os.environ.copy()
        env["GIT_TERMINAL_PROMPT"] = "0"
        token = env.get("GITHUB_TOKEN", "")
        if not token:
            return env, None
        handle = tempfile.NamedTemporaryFile("w", delete=False, prefix="git-askpass-")
        handle.write(
            "#!/bin/sh\n"
            "case \"$1\" in\n"
            "  *sername*) printf '%s\\n' \"$GITHUB_USERNAME\" ;;\n"
            "  *)         printf '%s\\n' \"$GITHUB_TOKEN\" ;;\n"
            "esac\n"
        )
        handle.close()
        os.chmod(handle.name, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
        env["GIT_ASKPASS"] = handle.name
        return env, handle.name

    auth_env, askpass_path = git_auth_env()

    def cmd(command, cwd=None, use_git_auth=False, extra_env=None):
        env = auth_env.copy() if use_git_auth else os.environ.copy()
        if extra_env:
            env.update(extra_env)
        subprocess.run(command, cwd=cwd, env=env, check=True)

    code = Path("/tmp/code")
    data = Path("/tmp/data")
    cmd(["git", "config", "--global", "user.name", "Kubeflow Pipeline"])
    cmd(["git", "config", "--global", "user.email", "kubeflow@localhost"])

    cmd(["git", "clone", code_repo_url, str(code)], use_git_auth=True)
    cmd(["git", "checkout", "--detach", code_commit], code, True)
    cmd(["datalad", "clone", dataset_repo_url, str(data)], use_git_auth=True)
    cmd(["git", "checkout", "--detach", dataset_commit], data, True)
    cmd(["datalad", "get", "-r", "."], data, True)

    venv = Path("/tmp/venv")
    cmd([sys.executable, "-m", "venv", str(venv)])
    runtime_python = str(venv / "bin" / "python")
    cmd([runtime_python, "-m", "pip", "install", "-r", str(code / "requirements.txt")])

    bundle_dir = Path("/tmp/manifest-bundle")
    bundle_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(manifest_bundle.path, "r") as archive:
        archive.extractall(bundle_dir)

    source_test = bundle_dir / "test_datalad.txt"
    test_manifest = Path("/tmp/test_datalad.txt")
    lines = []
    for raw in source_test.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        path, label = raw.split("\t")
        candidate = Path(path)
        resolved = candidate if candidate.is_absolute() else data / candidate
        lines.append(f"{resolved.resolve()}\t{label}")
    test_manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")

    mlflow.set_workspace(mlflow_workspace)
    model_path = Path(mlflow.artifacts.download_artifacts(artifact_uri=model_uri))
    checkpoints = list(model_path.rglob("*.pth")) if model_path.is_dir() else [model_path]
    if not checkpoints:
        raise FileNotFoundError(f"No .pth model file found in {model_path}")

    out = Path("/tmp/eval")
    execution_id = "eval-" + time.strftime("%Y%m%d%H%M%S")
    command = [
        runtime_python, "-u", "eval.py",
        "--id", execution_id,
        "--iut_paths_file", str(test_manifest),
        "--image_size", str(image_size),
        "--out_dir", str(out),
        "--model", model,
        "--load_path", str(checkpoints[0]),
        "--repo", dataset_repo_url,
        "--commit", dataset_commit,
        "--name", dataset_name,
        "--dataset_root", str(data),
        "--workspace", mlflow_workspace,
        "--experiment", mlflow_experiment,
    ]
    cmd(
        command,
        code,
        extra_env={
            "PARENT_MLFLOW_RUN_ID": parent_mlflow_run_id,
            "PIPELINE_KIND": pipeline_kind,
        },
    )

    result = json.loads((out / "result.json").read_text())
    if askpass_path:
        Path(askpass_path).unlink(missing_ok=True)
    return (float(result["accuracy"]), result["mlflow_run_id"])


def _configure_runtime_task(task):
    kubernetes.use_secret_as_env(
        task,
        secret_name="github-credentials",
        secret_key_to_env={
            "username": "GITHUB_USERNAME",
            "token": "GITHUB_TOKEN",
        },
    )

    task.set_env_variable(
        name="MLFLOW_TRACKING_URI",
        value=MLFLOW_TRACKING_URI,
    )
    task.set_env_variable(
        name="MLFLOW_TRACKING_USERNAME",
        value=MLFLOW_TRACKING_USERNAME,
    )
    task.set_env_variable(
        name="MLFLOW_TRACKING_PASSWORD",
        value=MLFLOW_TRACKING_PASSWORD,
    )

    kubernetes.set_image_pull_secrets(task, [IMAGE_PULL_SECRET])
    return task


def _workflow(
    *,
    pipeline_kind: str,
    code_repo_url: str,
    dataset_repo_url: str,
    manifest_bundle_uri: str,
    code_commit: str,
    dataset_commit: str,
    run_name: str,
    seed: int,
    model: str,
    image_size: int,
    batch_size: int,
    workers: int,
    optimizer: str,
    learning_rate: float,
    epochs: int,
    factor: float,
    patience: int,
    early_stopping_patience: int,
    n_c_samples: int,
    val_n_c_samples: int,
    load_model_uri: str,
    mlflow_workspace: str,
    mlflow_experiment: str,
):
    manifests = dsl.importer(
        artifact_uri=manifest_bundle_uri,
        artifact_class=Dataset,
        reimport=False,
    )

    train = train_local(
        code_repo_url=code_repo_url,
        dataset_repo_url=dataset_repo_url,
        code_commit=code_commit,
        dataset_commit=dataset_commit,
        manifest_bundle=manifests.output,
        pipeline_kind=pipeline_kind,
        run_name=run_name,
        seed=seed,
        model=model,
        image_size=image_size,
        batch_size=batch_size,
        workers=workers,
        optimizer=optimizer,
        learning_rate=learning_rate,
        epochs=epochs,
        factor=factor,
        patience=patience,
        early_stopping_patience=early_stopping_patience,
        n_c_samples=n_c_samples,
        val_n_c_samples=val_n_c_samples,
        load_model_uri=load_model_uri,
        mlflow_workspace=mlflow_workspace,
        mlflow_experiment=mlflow_experiment,
    ).set_display_name("Train model")
    _configure_runtime_task(train)

    evaluate = evaluate_local(
        code_repo_url=code_repo_url,
        dataset_repo_url=dataset_repo_url,
        code_commit=train.outputs["code_commit_resolved"],
        dataset_commit=train.outputs["dataset_commit_resolved"],
        dataset_name=train.outputs["dataset_name"],
        manifest_bundle=manifests.output,
        parent_mlflow_run_id=train.outputs["mlflow_run_id"],
        model_uri=train.outputs["model_uri"],
        pipeline_kind=pipeline_kind,
        model=model,
        image_size=image_size,
        mlflow_workspace=mlflow_workspace,
        mlflow_experiment=mlflow_experiment,
    ).set_display_name("Evaluate model")
    _configure_runtime_task(evaluate)


@dsl.pipeline(name="new-training-and-evaluation")
def new_training_pipeline(
    code_repo_url: str,
    dataset_repo_url: str,
    manifest_bundle_uri: str,
    code_commit: str = "",
    dataset_commit: str = "",
    run_name: str = "cpu-demo",
    seed: int = 3721,
    model: str = "ours",
    image_size: int = 512,
    batch_size: int = 1,
    workers: int = 0,
    optimizer: str = "adamw",
    learning_rate: float = 0.001,
    epochs: int = 2,
    factor: float = 0.9,
    patience: int = 5,
    early_stopping_patience: int = 10,
    n_c_samples: int = -1,
    val_n_c_samples: int = -1,
    load_model_uri: str = "",
    mlflow_workspace: str = "prefect-test",
    mlflow_experiment: str = "pipeline-new",
):
    _workflow(
        pipeline_kind="new",
        code_repo_url=code_repo_url,
        dataset_repo_url=dataset_repo_url,
        manifest_bundle_uri=manifest_bundle_uri,
        code_commit=code_commit,
        dataset_commit=dataset_commit,
        run_name=run_name,
        seed=seed,
        model=model,
        image_size=image_size,
        batch_size=batch_size,
        workers=workers,
        optimizer=optimizer,
        learning_rate=learning_rate,
        epochs=epochs,
        factor=factor,
        patience=patience,
        early_stopping_patience=early_stopping_patience,
        n_c_samples=n_c_samples,
        val_n_c_samples=val_n_c_samples,
        load_model_uri=load_model_uri,
        mlflow_workspace=mlflow_workspace,
        mlflow_experiment=mlflow_experiment,
    )


@dsl.pipeline(name="reproduce-training-and-evaluation")
def reproduce_training_pipeline(
    code_repo_url: str,
    dataset_repo_url: str,
    manifest_bundle_uri: str,
    code_commit: str,
    dataset_commit: str,
    run_name: str = "reproduce-cpu",
    seed: int = 3721,
    model: str = "ours",
    image_size: int = 512,
    batch_size: int = 1,
    workers: int = 0,
    optimizer: str = "adamw",
    learning_rate: float = 0.001,
    epochs: int = 2,
    factor: float = 0.9,
    patience: int = 5,
    early_stopping_patience: int = 10,
    n_c_samples: int = -1,
    val_n_c_samples: int = -1,
    load_model_uri: str = "",
    mlflow_workspace: str = "prefect-test",
    mlflow_experiment: str = "pipeline-new",
):
    _workflow(
        pipeline_kind="reproduce",
        code_repo_url=code_repo_url,
        dataset_repo_url=dataset_repo_url,
        manifest_bundle_uri=manifest_bundle_uri,
        code_commit=code_commit,
        dataset_commit=dataset_commit,
        run_name=run_name,
        seed=seed,
        model=model,
        image_size=image_size,
        batch_size=batch_size,
        workers=workers,
        optimizer=optimizer,
        learning_rate=learning_rate,
        epochs=epochs,
        factor=factor,
        patience=patience,
        early_stopping_patience=early_stopping_patience,
        n_c_samples=n_c_samples,
        val_n_c_samples=val_n_c_samples,
        load_model_uri=load_model_uri,
        mlflow_workspace=mlflow_workspace,
        mlflow_experiment=mlflow_experiment,
    )


@dsl.pipeline(name="retrain-and-evaluate")
def retrain_pipeline(
    code_repo_url: str,
    dataset_repo_url: str,
    manifest_bundle_uri: str,
    code_commit: str,
    dataset_commit: str = "",
    run_name: str = "retrain-cpu",
    seed: int = 3721,
    model: str = "ours",
    image_size: int = 512,
    batch_size: int = 1,
    workers: int = 0,
    optimizer: str = "adamw",
    learning_rate: float = 0.0001,
    epochs: int = 3,
    factor: float = 0.9,
    patience: int = 5,
    early_stopping_patience: int = 10,
    n_c_samples: int = -1,
    val_n_c_samples: int = -1,
    load_model_uri: str = "",
    mlflow_workspace: str = "prefect-test",
    mlflow_experiment: str = "pipeline-new",
):
    _workflow(
        pipeline_kind="retrain",
        code_repo_url=code_repo_url,
        dataset_repo_url=dataset_repo_url,
        manifest_bundle_uri=manifest_bundle_uri,
        code_commit=code_commit,
        dataset_commit=dataset_commit,
        run_name=run_name,
        seed=seed,
        model=model,
        image_size=image_size,
        batch_size=batch_size,
        workers=workers,
        optimizer=optimizer,
        learning_rate=learning_rate,
        epochs=epochs,
        factor=factor,
        patience=patience,
        early_stopping_patience=early_stopping_patience,
        n_c_samples=n_c_samples,
        val_n_c_samples=val_n_c_samples,
        load_model_uri=load_model_uri,
        mlflow_workspace=mlflow_workspace,
        mlflow_experiment=mlflow_experiment,
    )
