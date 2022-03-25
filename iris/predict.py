from typing import Dict, List

import numpy as np
import torch

from iris import data, train


def predict(
    texts: List, artifacts: Dict, device: torch.device = torch.device("cpu")
) -> Dict:
    """Predict iris for an input features from
    best model in the `best` experiment.



    Note:
        The input parameter `texts` can hold multiple input texts and so the resulting prediction dictionary will have `len(texts)` items.

    Args:
        texts (List): List of input parametes(texts format) to predict iris for.
        artifacts (Dict): Artifacts needed for inference.
        device (torch.device): Device to run model on. Defaults to CPU.

    Returns:
        Predicted iris for each of the input text parameters.

    """
    # Retrieve artifacts
    params = artifacts["params"]
    # label_encoder = artifacts["label_encoder"]
    # tokenizer = artifacts["tokenizer"]
    model = artifacts["model"]
    X = np.array([texts])
    # print(X)
    y_filler = np.zeros((len(X), 3))
    dataset = data.CSVDataset(X=X, y=y_filler)
    dataloader = dataset.get_dataloader(batch_size=int(params.batch_size))

    # Get predictions
    trainer = train.Trainer(model=model, device=device)
    _, y_prob = trainer.predict_step(dataloader)
    y_pred = np.argmax(y_prob, axis=1).tolist()
    X = X[0].tolist()
    predictions = [
        {
            "input_text": X[i],
            "predicted_tags": y_pred[i],
        }
        for i in range(len(y_pred))
    ]

    return predictions