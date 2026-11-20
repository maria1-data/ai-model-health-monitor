
"""Register the existing trained model in the Docker-local MLflow registry."""

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

from src.config import ROOT_DIR, load_config
from src.training.artifacts import load_model_artifact
from src.tracking.mlflow_setup import (
    ensure_experiment,
    resolve_artifact_location,
    resolve_tracking_uri,
    validate_tracking_config,
)


def main():
    config = load_config()
    tracking_config = config["tracking"]

    validate_tracking_config(tracking_config)

    tracking_uri = resolve_tracking_uri(tracking_config, ROOT_DIR)
    artifact_location = resolve_artifact_location(
        tracking_config, ROOT_DIR
    )

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    experiment_id = ensure_experiment(
        client,
        tracking_config["experiment_name"],
        artifact_location,
    )

    model_path = ROOT_DIR / "models" / "churn_pipeline_v1.joblib"
    pipeline, metadata = load_model_artifact(model_path)

    model_name = tracking_config["registry"]["model_name"]
    alias = tracking_config["registry"]["candidate_alias"]

    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name="docker_model_registration",
    ) as run:
        mlflow.set_tags({
            "model_type": metadata["model_name"],
            "model_version": metadata["model_version"],
            "registration_source": "existing_trained_artifact",
        })

        model_info = mlflow.sklearn.log_model(
            pipeline,
            name="model",
            registered_model_name=model_name,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )

        versions = client.search_model_versions(
            f"name='{model_name}'"
        )

        matching_versions = [
            version for version in versions
            if version.run_id == run.info.run_id
        ]

        if not matching_versions:
            raise RuntimeError(
                "MLflow registered the model, but its version "
                "could not be located."
            )

        registered_version = max(
            matching_versions,
            key=lambda version: int(version.version),
        )

        client.set_registered_model_alias(
            model_name,
            alias,
            registered_version.version,
        )

    print("Docker MLflow registration successful.")
    print(f"Model: {model_name}")
    print(f"Version: {registered_version.version}")
    print(f"Alias: {alias}")
    print(f"Tracking URI: {tracking_uri}")
    print(f"Artifact URI: {model_info.artifact_path}")


if __name__ == "__main__":
    main()
