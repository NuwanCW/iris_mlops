# tests/tagifai/test_data.py
# Test tagifai/data.py components.

import numpy as np
import pandas as pd
import pytest

from iris import data


def test_train_test_split():
    # Process
    path = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
    df = pd.read_csv(path, header=None, names=["f1", "f2", "f3", "f4", "class"])
    df = df.sample(frac=1).reset_index(drop=True)
    # df["class"] = LabelEncoder().fit_transform(df["class"])
    train_df = df[: int(len(df) * 0.8)].reset_index(drop=True)
    test_df = df[int(len(df) * 0.8) : int(len(df) * 0.9)].reset_index(drop=True)
    val_df = df[int(len(df) * 0.9) :].reset_index(drop=True)
    X_train, y_train = train_df.values[:, :-1], train_df.values[:, -1]
    X_val, y_val = val_df.values[:, :-1], val_df.values[:, -1]
    X_test, y_test = test_df.values[:, :-1], test_df.values[:, -1]

    assert len(X_train) == len(y_train)
    assert len(X_val) == len(y_val)
    assert len(X_test) == len(y_test)
    assert len(X_train) / float(len(df)) == pytest.approx(0.8, abs=0.05)  # 0.8 ± 0.05
    assert len(X_val) / float(len(df)) == pytest.approx(0.1, abs=0.05)  # 0.1 ± 0.05
    assert len(X_test) / float(len(df)) == pytest.approx(0.1, abs=0.05)  # 0.1 ± 0.05


class TestCSVDataset:
    def setup_method(self):
        """Called before every method."""
        self.X = np.array([[4, 2, 3, 0], [2, 4, 3, 3], [2, 3, 0, 0]])
        self.y = np.array([[1], [2], [0]])
        self.batch_size = 1
        self.dataset = data.CSVDataset(X=self.X, y=self.y)

    def teardown_method(self):
        """Called after every method."""
        del self.dataset

    def test_len(self):
        assert len(self.X) == len(self.dataset)

    def test_str(self):
        assert str(self.dataset) == f"<Dataset(N={len(self.dataset)})>"
