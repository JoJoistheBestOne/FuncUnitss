import os
import sys
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)

class ToyDataset(Dataset):
    def __len__(self):
        return 100

    def __getitem__(self, idx):
        return torch.rand(10), torch.rand(10)

def setup(rank, world_size):
    # os.environ['MASTER_ADDR'] = '172.17.0.2'
    # os.environ['MASTER_PORT'] = '50574'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):

    setup(rank, world_size)

    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    dataset = ToyDataset()
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=10)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    for epoch in range(2):  # loop over the dataset multiple times
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(rank)
            labels = labels.to(rank)

            optimizer.zero_grad()
            outputs = ddp_model(inputs)
            loss = criterion(outputs, labels)
            print(f"rank: {rank}, epoch: {epoch}, iteration:{i}, loss: {loss.item()}")
            loss.backward()
            optimizer.step()

    cleanup()

def main():
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    train(rank, world_size)

if __name__ == "__main__":
    main()