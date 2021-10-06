#!/usr/bin/python3

# this is a temporary hack to allow importing the files in the TreeLoss folder when the library is not yet installed
import sys
sys.path.append('../..')

import logging
import os
from collections import defaultdict
################################################################################
# command line arguments
################################################################################
logging.info('processing command line arguments')
import argparse
parser = argparse.ArgumentParser(description='synthetic data experiment')

parser_data = parser.add_argument_group(
        title='data generation',
        description='the variable names for these arguments match exactly the variable names used in the paper'
        )
parser_data.add_argument('--exp_num', type=int, default=1)
parser_data.add_argument('--a', type=int, default=5)
parser_data.add_argument('--c', type=int, default=1000)
parser_data.add_argument('--n', type=int, default=10000)
parser_data.add_argument('--d', type=int, default=1000)
parser_data.add_argument('--sigma', type=float, default=1.0)
parser_data.add_argument('--random', type=float, default=0.0)
parser_data.add_argument('--seed', type=int, default=666)
parser_data.add_argument('--batch', type=int, default=1000)
parser_data.add_argument('--epoch', type=int, default=10)

parser_model = parser.add_argument_group(
        title='model hyperparameters',
        description="it's recommended to always use scientific notation for hyperparameter values since we really care about the order of magnitude",
        )
parser_model.add_argument('--lr', type=float, default=1e-4, metavar='LR') 
parser_model.add_argument('--momentum', type=float, default=0.9)
parser_model.add_argument('--weight_decay', type=float, default=3e-4)
parser_model.add_argument('--loss', choices=['tree','xentropy','simloss', 'HSM'], default='xentropy')
parser.add_argument('--lower_bound', type=float, default=0.5)

parser_debug = parser.add_argument_group(title='debug')
parser_model.add_argument('--logdir', default='log')
args = parser.parse_args()

# load imports;
# we do this after parsing the command line arguments because it takes a long time,
# and we want immediate feedback from the command line parser if we have an invalid argument or pass the --help option
logging.info('load imports')
import random
import numpy as np
import math
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from TreeLoss.cover_tree import CoverTree
from TreeLoss.utilities import set_seed, gen_sim, level
from TreeLoss.loss import CoverTreeLoss, SimLoss, HSM

# set the seed
logging.debug('set_seed('+str(args.seed)+')')
set_seed(args.exp_num*10)

U = np.random.normal(size=[args.c, args.a])
V = np.random.normal(size=[args.a, args.d])
W_star = U @ V


# generate the data
Y = np.random.choice(range(args.c), size=[args.n])


X = []
for i in range(args.n):
    x_i = np.random.normal(W_star[Y[i],:], args.sigma)
    X.append(x_i)
X = np.array(X)

Y = torch.LongTensor(Y)
X = torch.Tensor(X)
logging.debug("Y.shape="+str(Y.shape))
logging.debug("X.shape="+str(X.shape))

new2index, length, tree = CoverTreeLoss.tree_structure(args.c,U, base=2)
level_ = level(tree)
height = -level_

class Mod(torch.nn.Module):
    def __init__(self, class_num, dimension, true_classnum, new2index, epsilon: float = 1e-8) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.fc = nn.Linear(dimension, class_num, bias=False)
        self.criterion = CoverTreeLoss(true_classnum, class_num, dimension, new2index)

    def forward(self,
                x: torch.Tensor, #(batch_size, hidden_size)
                y: torch.Tensor) -> torch.Tensor:

        weight = self.fc.weight
        loss, logits, added_weights = self.criterion(weight, x, y)
        return loss, logits, weight

model = Mod(length, args.d, args.c, new2index).cuda()
def get_levelnode(tree):
    level_dict = defaultdict(list)
    def pre_order(root, level):
        if not root:
            return
        level_dict[level].append(root.ctr_idx)
        if not isinstance(root, CoverTree._LeafNode):
            for child in root.children:
                pre_order(child, level+1)

    pre_order(tree.root, 0)
    return level_dict     
# level_dict = get_levelnode(tree)

optimizer = optim.SGD(
    model.parameters(),
    lr=args.lr,
    momentum=args.momentum,
    weight_decay=args.weight_decay,
    )
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

correct = 1e-10

for ep in range(args.epoch):
    for start in range(0, args.n, args.batch):
        train_X = torch.FloatTensor(X[start:start+args.batch].view(args.batch,args.d))
        train_Y = torch.LongTensor(Y[start:start+args.batch].view(args.batch))
        train_X, train_Y = train_X.cuda(), train_Y.cuda()
        loss, logits, W = model(train_X, train_Y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

level_list = [set() for _ in range(height)]
for leaf, path in new2index.items():
    for i, node in enumerate(path):
        level_list[i].add(node)

# W_norm = []
# for i in level_list:
#     leng = len(i)
#     node = torch.LongTensor(list(i))
#     W_norm.append(torch.norm(W[node,:])/leng)

W_norm = []
for i in level_list:
    leng = len(i)
    node = torch.LongTensor(list(i))
    temp = 0
    for j in node:
        temp += torch.linalg.norm(W[j,:])
    W_norm.append(temp/leng)

print(W_norm)

