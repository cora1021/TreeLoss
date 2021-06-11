import numpy as np
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/sythetic_data/num_data')
# writer = SummaryWriter('runs/sythetic_data/dimension')
# writer = SummaryWriter('runs/sythetic_data/sigma')

import argparse
parser = argparse.ArgumentParser(description='Synthetic data')
parser.add_argument('--num_data', type=int, default=100000)                      
parser.add_argument('--dimension', type=int, default=64)
parser.add_argument('--seed', type=int, default=666)
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR')
args = parser.parse_args()
from utilities import set_seed

set_seed(args)

class Loss(torch.nn.Module):
    def __init__(self, class_num, dimension, epsilon: float = 1e-8) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.fc = nn.Linear(dimension, class_num, bias=False)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self,
                x: torch.Tensor, #(batch_size, hidden_size)
                y: torch.Tensor) -> torch.Tensor:

        logits = self.fc(x)
        
        loss = self.criterion(logits, y)
        return loss

# loss vs number of data points
means = np.zeros((100, args.dimension))
stds = np.ones((100, args.dimension))
w = np.random.normal(means, stds)
sigma = np.random.rand(1, args.dimension)
synthetic_loss = Loss(100, args.dimension)
optimizer = optim.SGD(synthetic_loss.parameters(), lr=args.lr,
                        momentum=0.9, weight_decay=3e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100000)
for i in range(1, args.num_data+1):
    y = np.random.randint(0, 100)
    x = torch.FloatTensor(np.random.normal(w[y,:], sigma))
    target = torch.LongTensor([y])
    loss = synthetic_loss(x, target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step() 
    writer.add_scalar('Loss vs number of data points: ', loss, i)
    writer.add_scalar('Loss vs number of data points(Log): ', math.log(loss), math.log(i))

#  loss vs dimension
# dimension = [8,16,32,64,128,256,512,1024]
# for dim in dimension:
#     means = np.zeros((100, dim))
#     stds = np.ones((100, dim))
#     w = np.random.normal(means, stds)
#     sigma = np.random.rand(1, dim)
#     synthetic_loss = Loss(100, dim)
#     optimizer = optim.SGD(synthetic_loss.parameters(), lr=args.lr,
#                             momentum=0.9, weight_decay=3e-4)
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100000)
#     for i in range(1, args.num_data+1):
#         y = np.random.randint(0, 100)
#         x = torch.FloatTensor(np.random.normal(w[y,:], sigma))
#         target = torch.LongTensor([y])
#         loss = synthetic_loss(x, target)
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()
#         scheduler.step() 
#     writer.add_scalar('Loss vs dimension ', loss, dim)
#     writer.add_scalar('Loss vs dimension(Log) ', math.log(loss), math.log(dim))

# loss vs sigma
# for sig in tqdm(range(1,101)):
#     means = np.zeros((100, args.dimension))
#     stds = np.ones((100, args.dimension))
#     w = np.random.normal(means, stds)
#     sigma = np.full((1, args.dimension), sig)
#     synthetic_loss = Loss(100, args.dimension)
#     optimizer = optim.SGD(synthetic_loss.parameters(), lr=args.lr,
#                             momentum=0.9, weight_decay=3e-4)
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100000)
#     for i in range(1, args.num_data+1):
#         y = np.random.randint(0, 100)
#         x = torch.FloatTensor(np.random.normal(w[y,:], sigma))
#         target = torch.LongTensor([y])
#         loss = synthetic_loss(x, target)
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()
#         scheduler.step() 
#     writer.add_scalar('Loss vs sigma ', loss, sig)
#     writer.add_scalar('Loss vs sigma(Log) ', math.log(loss), math.log(sig))
