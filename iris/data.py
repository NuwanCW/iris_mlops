# iris/data.py
# Data processing

import json
from argparse import Namespace
from pathlib import Path
from re import X
from typing import List
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from config import config
from iris import utils
from torch.utils.data import DataLoader, Dataset

TRAIN_SIZ = 0.3


class CSVDataset(Dataset):
    """Create dataloader object for efficient data feeding
    usage:
        dataset = CSVDataset(X=X,y=y)
    """

    def __init__(self, X, y):
        self.X = torch.LongTensor(X.astype(np.float32))
        self.y = torch.LongTensor(y.astype(np.int32))

    def __len__(self):
        return len(self.y)

    def __str__(self):
        return f"<Dataset(N={len(self)})>"

    def __getitem__(self, index: int) -> List:
        X = self.X[index]
        y = self.y[index]
        return [X, y]

    def get_dataloader(
        self, batch_size: int, shuffle: bool = False, drop_last: bool = False
    ) -> torch.utils.data.DataLoader:
        return DataLoader(
            dataset=self,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            pin_memory=True,
        )