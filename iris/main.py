import json
import os
import tempfile
import warnings
from argparse import Namespace

# from datetime import datetime
from pathlib import Path
from typing import Dict  # , Optional

import mlflow
import optuna

# import pandas as pd
import torch
import typer
from mlflow.tracking import MlflowClient

# from feast import FeatureStore
from numpyencoder import NumpyEncoder
from optuna.integration.mlflow import MLflowCallback

from config import config
from config.config import logger
from iris import data, models, predict, train, utils

# # Ignore warning
warnings.filterwarnings("ignore")
# Initialize Typer CLI app
app = typer.Typer()


@app.command()
def optimize(
    params_fp: Path = Path(config.CONFIG_DIR, "params.json"),
    study_name="optimization",
    num_trials=5,
):
    # Parameters
    params = Namespace(**utils.load_dict(filepath=params_fp))

    # Optimize
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    study = optuna.create_study(study_name=study_name, direction="maximize", pruner=pruner)
    mlflow_callback = MLflowCallback(tracking_uri=mlflow.get_tracking_uri(), metric_name="f1")
    study.optimize(
        lambda trial: train.objective(params, trial),
        n_trials=num_trials,
        callbacks=[mlflow_callback],
    )

    # All trials
    trials_df = study.trials_dataframe()
    trials_df = trials_df.sort_values(["value"], ascending=False)

    # Best trial
    logger.info(f"Best value (f1): {study.best_trial.value}")
    params = {**params.__dict__, **study.best_trial.params}
    print(json.dumps(params, indent=2, cls=NumpyEncoder))
    print(f"Best value (f1): {study.best_trial.value}")
    utils.save_dict(params, params_fp, cls=NumpyEncoder)
    logger.info(json.dumps(params, indent=2, cls=NumpyEncoder))


@app.command()
def train_model(
    params_fp: Path = Path(config.CONFIG_DIR, "params.json"),
    experiment_name="best",
    run_name="model",
    test_run=False,
) -> None:
    # Parameters
    params = Namespace(**utils.load_dict(filepath=params_fp))

    # Start run
    mlflow.set_experiment(experiment_name=experiment_name)
    with mlflow.start_run(run_name=run_name):
        run_id = mlflow.active_run().info.run_id
        print(f"Run ID: {run_id}")
        # Train
        artifacts = train.train(params=params)

        # Set tags
        tags = {}
        mlflow.set_tags(tags)

        # Log metrics
        performance = artifacts["performance"]
        logger.info(json.dumps(performance["overall"], indent=2))
        print(json.dumps(performance["overall"], indent=2))
        metrics = {
            "precision": performance["overall"]["precision"],
            "recall": performance["overall"]["recall"],
            "f1": performance["overall"]["f1"],
            "best_val_loss": artifacts["loss"],
        }
        mlflow.log_metrics(metrics)

        # Log artifacts
        with tempfile.TemporaryDirectory() as dp:
            utils.save_dict(vars(artifacts["params"]), Path(dp, "params.json"), cls=NumpyEncoder)
            utils.save_dict(performance, Path(dp, "performance.json"))
            torch.save(artifacts["model"].state_dict(), Path(dp, "model.pt"))
            mlflow.log_artifacts(dp)
        mlflow.log_params(vars(artifacts["params"]))
        if not test_run:  # pragma: no cover, testing shouldn't save files
            open(Path(config.CONFIG_DIR, "run_id.txt"), "w").write(run_id)
            utils.save_dict(performance, Path(config.CONFIG_DIR, "performance.json"))


@app.command()
def predict_iris(text, run_id):
    # Predict
    artifacts = load_artifacts(run_id=run_id)
    text = [[float(i) for i in t.split(",")] for t in [text]]
    prediction = predict.predict(texts=text, artifacts=artifacts)
    print(prediction)
    logger.info(json.dumps(prediction, indent=2))
    return prediction


@app.command()
def params(run_id):
    params = vars(load_artifacts(run_id=run_id, best_f1=False)["params"])
    logger.info(json.dumps(params, indent=2))
    return params


@app.command()
def performance(run_id):
    performance = load_artifacts(run_id=run_id, best_f1=False)["performance"]
    logger.info(json.dumps(performance, indent=2))
    return performance


def load_artifacts(run_id: str, device: torch.device = torch.device("cpu"), best_f1=True) -> Dict:
    """Load artifacts for current model.

    Args:
        run_id (str): ID of the model run to load artifacts.
        device (torch.device): Device to run model on. Defaults to CPU.

    Returns:
        Artifacts needed for inference.
    """
    # Load artifacts
    if best_f1:
        experiment_id = mlflow.get_experiment_by_name("best").experiment_id
        all_runs = mlflow.search_runs(
            experiment_ids=experiment_id,
            order_by=["metrics.f1 DESC"],
        )
        run_id = all_runs.iloc[0].run_id
    else:
        run_id = run_id
    # artifact_uri = mlflow.get_run(run_id=run_id).info.artifact_uri
    client = MlflowClient()
    local_dir = "/tmp/artifact_downloads"
    if not os.path.exists(local_dir):
        os.mkdir(local_dir)
    client.download_artifacts(run_id, "params.json", local_dir)
    client.download_artifacts(run_id, "model.pt", local_dir)
    client.download_artifacts(run_id, "performance.json", local_dir)

    params = Namespace(**utils.load_dict(filepath=Path(local_dir, "params.json")))
    model_state = torch.load(Path(local_dir, "model.pt"), map_location=device)
    performance = utils.load_dict(filepath=Path(local_dir, "performance.json"))

    # Initialize model
    model = models.initialize_model()
    model.load_state_dict(model_state)
    # print(params)
    return {
        "params": params,
        "model": model,
        "performance": performance,
    }


def delete_experiment(experiment_name):
    client = mlflow.tracking.MlflowClient()
    experiment_id = client.get_experiment_by_name(experiment_name).experiment_id
    client.delete_experiment(experiment_id=experiment_id)
    logger.info(f"✅ Deleted experiment {experiment_name}!")
