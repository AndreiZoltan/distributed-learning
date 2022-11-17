import logging
import numpy as np

import torch

from pathlib import Path
from typing import Dict

from dc_framework.data_preparation import Dataset

logger = logging.getLogger("__name__")


def init(model: torch.nn.Module, criterion: torch.nn.Module):
    return DCFramework(model, criterion)


class DCFramework:
    def __init__(self, model: torch.nn.Module, criterion: torch.nn.Module, lr=1e-3):
        self.model = model
        self.optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        self.criterion = criterion

    def forward(self, feature, target):
        try:
            output = self.model(feature)
        except:
            logger.warning(f"feature: {feature}")
            raise
        try:
            loss = self.criterion(output, target)
        except:
            logger.warning(f"output: {output}")
            logger.warning(f"target: {target}")
            raise
        return {
            "output": output,
            "loss": loss
        }

    def train(self, train_data: Dict[str, np.array], batch_size: int = 1):
        train_data = Dataset(train_data)
        train_dataloader = train_data.get_dataloader(batch_size=batch_size)
        
        for batch in train_dataloader:
            output = self.forward(*batch)
            loss = output["loss"]
            loss.backward()
            self.optimizer.step()

    def check_accuracy(self, test_data: Dict[str, np.array], batch_size: int = 1, device: str = "cpu"):
        test_data = Dataset(test_data)
        test_dataloader = test_data.get_dataloader(batch_size=batch_size)

        num_correct = 0
        num_samples = 0
        self.model.eval()

        with torch.no_grad():
            for x, y in test_dataloader:
                x = x.to(device=device)
                y = y.to(device=device)

                scores = self.model(x)
                _, predictions = scores.max(1)
                num_correct += (predictions == y).sum()
                num_samples += predictions.size(0)

            print(
                f'Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}')

        self.model.train()

    def save(self, path: Path):
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state, path)

    def load(self, path: Path, device):
        state = torch.load(path, map_location=device)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
