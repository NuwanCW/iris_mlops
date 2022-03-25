import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support

from iris import data, train

# from snorkel.slicing import PandasSFApplier, slicing_function


def get_metrics(
    y_true,
    y_pred,
):
    # Performance
    metrics = {"overall": {}}

    # Overall metrics
    overall_metrics = precision_recall_fscore_support(
        y_true, y_pred, average="weighted"
    )
    metrics["overall"]["precision"] = overall_metrics[0]
    metrics["overall"]["recall"] = overall_metrics[1]
    metrics["overall"]["f1"] = overall_metrics[2]
    metrics["overall"]["num_samples"] = np.float64(len(y_true))
    return metrics


def evaluate(df, artifacts, device=torch.device("cpu")):
    # Artifacts
    params = artifacts["params"]
    model = artifacts["model"]
    # label_encoder = artifacts["label_encoder"]
    model = model.to(device)
    X_test, y_test = df.values[:, :-1], df.values[:, -1]
    test_dataset = data.CSVDataset(X=X_test, y=y_test)
    test_dataloader = test_dataset.get_dataloader(batch_size=params.batch_size)

    # Determine predictions using threshold
    trainer = train.Trainer(model=model, device=device)
    y_true, y_prob = trainer.predict_step(dataloader=test_dataloader)
    # print(y_prob.shape, y_prob)
    y_pred = np.argmax(y_prob, axis=1)

    # Evaluate performance
    performance = {}
    performance = get_metrics(y_true=y_true, y_pred=y_pred)

    return y_true, y_pred, performance
