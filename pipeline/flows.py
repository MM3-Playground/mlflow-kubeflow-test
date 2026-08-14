import os

from kfp import dsl, kubernetes
from kfp.dsl import Dataset


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} must be exported before compiling the pipeline")
    return value


RUNTIME_IMAGE = _required_env("KUBEFLOW_RUNTIME_IMAGE")
IMAGE_PULL_SECRET = _required_env("KUBEFLOW_IMAGE_PULL_SECRET")

MLFLOW_TRACKING_URI = _required_env("MLFLOW_TRACKING_URI")
MLFLOW_TRACKING_USERNAME = _required_env("MLFLOW_TRACKING_USERNAME")
MLFLOW_TRACKING_PASSWORD = _required_env("MLFLOW_TRACKING_PASSWORD")

# Used only to read the existing manifest bundle URI directly, avoiding dsl.importer/MLMD.
# These defaults match the current Kubeflow installation and can be overridden locally.
KFP_S3_ENDPOINT = os.environ.get(
    "KFP_S3_ENDPOINT", "https://object-arbutus.alliancecan.ca"
)
KFP_S3_REGION = os.environ.get("KFP_S3_REGION", "us-east-1")
KFP_S3_SECRET = os.environ.get("KFP_S3_SECRET", "kfp-s3-credentials")


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
def upload_manifest_bundle_pipeline(
    train_b64: str,
    val_b64: str,
    test_b64: str,
):
    write_manifest_bundle(
        train_b64=train_b64,
        val_b64=val_b64,
        test_b64=test_b64,
    )



@dsl.container_component
def train_task(
    code_repo_url: str, code_commit: str, dataset_repo_url: str, dataset_commit: str,
    manifest_bundle_uri: str, pipeline_kind: str, run_name: str,
    seed: int, model: str, image_size: int, batch_size: int, workers: int,
    optimizer: str, learning_rate: float, epochs: int, factor: float, patience: int,
    early_stopping_patience: int, n_c_samples: int, val_n_c_samples: int,
    load_model_uri: str, mlflow_workspace: str, mlflow_experiment: str,
    mlflow_run_id: dsl.OutputPath(str), model_uri: dsl.OutputPath(str),
    code_commit_resolved: dsl.OutputPath(str),
    dataset_commit_resolved: dsl.OutputPath(str), dataset_name: dsl.OutputPath(str),
):
    return dsl.ContainerSpec(
        image=RUNTIME_IMAGE,
        command=["/workspace/venv/bin/python", "-u", "-m", "pipeline.kubeflow_task"],
        args=[
            "train",
            "--code-repo-url", code_repo_url,
            "--code-commit", code_commit,
            "--dataset-repo-url", dataset_repo_url, "--dataset-commit", dataset_commit,
            "--manifest-bundle-uri", manifest_bundle_uri, "--pipeline-kind", pipeline_kind,
            "--run-name", run_name, "--seed", seed, "--model", model,
            "--image-size", image_size, "--batch-size", batch_size, "--workers", workers,
            "--optimizer", optimizer, "--learning-rate", learning_rate, "--epochs", epochs,
            "--factor", factor, "--patience", patience,
            "--early-stopping-patience", early_stopping_patience,
            "--n-c-samples", n_c_samples, "--val-n-c-samples", val_n_c_samples,
            "--load-model-uri", load_model_uri, "--mlflow-workspace", mlflow_workspace,
            "--mlflow-experiment", mlflow_experiment,
            "--out-mlflow-run-id", mlflow_run_id, "--out-model-uri", model_uri,
            "--out-code-commit", code_commit_resolved,
            "--out-dataset-commit", dataset_commit_resolved,
            "--out-dataset-name", dataset_name,
        ],
    )


@dsl.container_component
def evaluate_task(
    code_repo_url: str, code_commit: str, dataset_repo_url: str, dataset_commit: str,
    dataset_name: str, manifest_bundle_uri: str,
    parent_mlflow_run_id: str, model_uri: str, pipeline_kind: str, model: str,
    image_size: int, mlflow_workspace: str, mlflow_experiment: str,
    accuracy: dsl.OutputPath(float), mlflow_run_id: dsl.OutputPath(str),
):
    return dsl.ContainerSpec(
        image=RUNTIME_IMAGE,
        command=["/workspace/venv/bin/python", "-u", "-m", "pipeline.kubeflow_task"],
        args=[
            "evaluate",
            "--code-repo-url", code_repo_url,
            "--code-commit", code_commit,
            "--dataset-repo-url", dataset_repo_url, "--dataset-commit", dataset_commit,
            "--dataset-name", dataset_name, "--manifest-bundle-uri", manifest_bundle_uri,
            "--parent-mlflow-run-id", parent_mlflow_run_id, "--model-uri", model_uri,
            "--pipeline-kind", pipeline_kind, "--model", model, "--image-size", image_size,
            "--mlflow-workspace", mlflow_workspace, "--mlflow-experiment", mlflow_experiment,
            "--out-accuracy", accuracy, "--out-mlflow-run-id", mlflow_run_id,
        ],
    )


def _configure_runtime_task(task, code_repo_url, code_commit):
    # code_repo_url/code_commit are intentionally carried in the component args.
    # runtime-bootstrap.yaml reads them from the resolved Pod args.
    task.set_env_variable(name="PYTHONPATH", value="/workspace/code")

    # Read the manifest bundle directly from the KFP S3 object store.
    # This removes the dsl.importer task and its MLMD importer execution.
    kubernetes.use_secret_as_env(
        task,
        secret_name=KFP_S3_SECRET,
        secret_key_to_env={
            "accesskey": "AWS_ACCESS_KEY_ID",
            "secretkey": "AWS_SECRET_ACCESS_KEY",
        },
    )
    task.set_env_variable(name="AWS_ENDPOINT_URL", value=KFP_S3_ENDPOINT)
    task.set_env_variable(name="AWS_REGION", value=KFP_S3_REGION)
    task.set_env_variable(name="AWS_DEFAULT_REGION", value=KFP_S3_REGION)

    # Keep Git credentials as a Kubernetes Secret for now.
    # This can be changed later without changing train_local.py/eval.py.
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

    # This is the only opt-in required for the cluster-side bootstrap policy.
    kubernetes.add_pod_label(
        task,
        label_key="bootstrap",
        label_value="true",
    )

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
    train = train_task(
        code_repo_url=code_repo_url,
        code_commit=code_commit,
        dataset_repo_url=dataset_repo_url,
        dataset_commit=dataset_commit,
        manifest_bundle_uri=manifest_bundle_uri,
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
    _configure_runtime_task(train, code_repo_url, code_commit)

    evaluate = evaluate_task(
        code_repo_url=code_repo_url,
        code_commit=train.outputs["code_commit_resolved"],
        dataset_repo_url=dataset_repo_url,
        dataset_commit=train.outputs["dataset_commit_resolved"],
        dataset_name=train.outputs["dataset_name"],
        manifest_bundle_uri=manifest_bundle_uri,
        parent_mlflow_run_id=train.outputs["mlflow_run_id"],
        model_uri=train.outputs["model_uri"],
        pipeline_kind=pipeline_kind,
        model=model,
        image_size=image_size,
        mlflow_workspace=mlflow_workspace,
        mlflow_experiment=mlflow_experiment,
    ).set_display_name("Evaluate model")
    _configure_runtime_task(evaluate, code_repo_url, train.outputs["code_commit_resolved"])


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
    mlflow_workspace: str = "kubeflow-test",
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
    mlflow_workspace: str = "kubeflow-test",
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
    mlflow_workspace: str = "kubeflow-test",
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