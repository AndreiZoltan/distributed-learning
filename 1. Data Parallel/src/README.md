# Distributed Data Parallel

Привет! Тут находятся примеры, которые показывают, как использовать распределенные вычисления в PyTorch.

Для запуска `train.py` можно использовать обычный python:
```
python train.py
```

Для запуска `train_distributed.py` следует использовать:
```
torchrun --nnodes 1 --nproc_per_node=2 train_distributed.py
```
