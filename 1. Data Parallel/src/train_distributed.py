import torch
import os

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import (
    get_rank,
    init_process_group,
)

from model import Net


def get_dataloader():
    dataset = MNIST(
        "./mnist",
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    )
    sampler = DistributedSampler(dataset)
    return DataLoader(dataset, sampler=sampler, batch_size=16)


def train():
    world_size = int(os.environ["WORLD_SIZE"])
    loader = get_dataloader()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    epoch_loss = torch.Tensor([0.0]).to(device)
    for epoch in range(10):
        loader.sampler.set_epoch(epoch)
        for steps, (data, target) in enumerate(loader):
            optimizer.zero_grad()
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            loss = torch.nn.functional.cross_entropy(output, target)
            epoch_loss += loss.item() / world_size
            loss.backward()
            optimizer.step()

            if steps % 200 == 0 and steps > 0:
                torch.distributed.all_reduce(
                    epoch_loss, op=torch.distributed.ReduceOp.SUM
                )
                if local_rank == 0:
                    print(
                        f"Epoch: {epoch}, Step: {steps}, Loss: {epoch_loss.item()/steps:.6f}"
                    )
            epoch_loss *= 0


if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    init_process_group("nccl")
    device = f"cuda:{local_rank}"
    model = Net().to(device)
    torch.cuda.set_device(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], output_device=local_rank
    )

    train()
