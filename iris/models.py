from argparse import Namespace
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


class ANN_model(nn.Module):
    # define model eliments
    def __init__(self, input_dim=4, hidden_dim=10, num_classes=3):
        super(ANN_model, self).__init__()
        # input to first hidden layer
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, X):
        z = F.relu(self.fc1(X.float()))
        z = self.fc2(z)
        return z


# initialize model
def initialize_model(device=torch.devcie("cpu")):
    model = ANN_model(
        input_dim=4,
        hidden_dim=10,
        num_classes=3,
    )
    model = model.to(device)
    return model