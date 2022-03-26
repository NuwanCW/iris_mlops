# tests/iris/test_eval.py
# Test iris/eval.py components.

import numpy as np

from iris import eval


def test_get_metrics():
    y_true = np.array([[1], [1], [2], [2], [0]])
    y_pred = np.array([[0], [1], [2], [1], [0]])
    performance = eval.get_metrics(y_true=y_true, y_pred=y_pred)
    # assert performance["overall"]["precision"] == (3 / 1 + 2 / 5) / 2
    # assert performance["overall"]["recall"] == (1 / 2 + 2 / 2) / 2
    # assert performance["overall"]["f1"] == (1 / 2 + 2 / 2) / 2
    assert performance["overall"]["num_samples"] == np.float64(len(y_true))
