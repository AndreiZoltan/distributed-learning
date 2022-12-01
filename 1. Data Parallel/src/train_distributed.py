import argparse

import torch
from tqdm.auto import tqdm

import subprocess
from time import sleep

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torch.utils.data.distributed import DistributedSampler
from torch.cuda import device_count
from torch.distributed import (
    get_rank,
    get_world_size,
    all_reduce,
    ReduceOp,
    init_process_group,
)

from model import Net


def get_dataloader(is_distr):
    dataset = MNIST(
        "./mnist",
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    )
    if is_distr:
        sampler = DistributedSampler(dataset)
        return DataLoader(dataset, sampler=sampler, batch_size=16)
    else:
        return DataLoader(dataset, batch_size=16)


def train(device, is_distr):
    # device = f"cuda:{torch.distributed.get_rank()}"
    world_size = 1
    if is_distr:
        world_size = get_world_size()
    model = Net()

    loader = get_dataloader(is_distr)
    model.to(device)
    if is_distr:
        model = torch.nn.parallel.DistributedDataParallel(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    steps = 0
    epoch_loss = 0
    loader_range = tqdm(loader) if args.local_rank == 0 else loader
    for data, target in loader_range:
        optimizer.zero_grad()
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        loss = torch.nn.functional.cross_entropy(output, target) / world_size
        epoch_loss += loss.item()
        if is_distr:
            all_reduce(loss, op=ReduceOp.SUM)
        loss.backward()
        optimizer.step()
        steps += 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--world_size", default=2)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    devices = device_count()
    is_distr = False
    local_rank = 0
    if devices == 0:
        device = torch.device("cpu")
    elif devices == 1:
        device = torch.device("cuda")
    else:
        init_process_group(
            "nccl",
            rank=local_rank,
            world_size=min(args.world_size, devices),
        )
        device = f"cuda:{get_rank()}"
        is_distr = True
    train(device, is_distr)
