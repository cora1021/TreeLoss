
import sys
sys.path.append('../..')

import logging
import os
import argparse
parser = argparse.ArgumentParser(description='synthetic data experiment')

parser_data = parser.add_argument_group(
        title='data generation',
        description='the variable names for these arguments match exactly the variable names used in the paper'
        )
parser_data.add_argument('--exp_num', type=int, default=1)
parser_data.add_argument('--a', type=int, default=5)
parser_data.add_argument('--c', type=int, default=100)
parser_data.add_argument('--n', type=int, default=1000)
parser_data.add_argument('--d', type=int, default=64)
parser_data.add_argument('--sigma', type=float, default=1.0)
parser_data.add_argument('--seed', type=int, default=666)
parser_data.add_argument('--test', type=int, default=10000)

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
parser_model.add_argument('--experiment', choices=['loss_vs_n','loss_vs_d', 'loss_vs_sigma', 'loss_vs_c', 'loss_vs_structure'], required=True)
args = parser.parse_args()

import random
import numpy as np
import math
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from TreeLoss.utilities import set_seed, gen_sim
from TreeLoss.loss import CoverTreeLoss, SimLoss, HSM

set_seed(args.exp_num*10)

U = np.random.normal(size=[args.c, args.a])
V = np.random.normal(size=[args.a, args.d])
W_star = U @ V

# projected 
A = np.random.normal(size=[args.d, 5])
u, s, vh = np.linalg.svd(A, full_matrices=True)
proj_matrix = u*u.T
R = proj_matrix[:, :5]

W_proj = W_star @ R

# training set
Y = np.random.choice(range(args.c), size=[args.n])
X = []
for i in range(args.n):
    x_i = np.random.normal(W_star[Y[i],:], args.sigma)
    X.append(x_i)
X = np.array(X)

Y = torch.LongTensor(Y)
X = torch.Tensor(X)

# testing set
Y_ = np.random.choice(range(args.c), size=[args.test])
X_ = []
for i in range(args.test):
    x_i = np.random.normal(W_star[Y_[i],:], args.sigma)
    X_.append(x_i)
X_ = np.array(X_)

Y_ = torch.LongTensor(Y_)
X_ = torch.Tensor(X_)

if args.loss == 'xentropy':
    model = nn.Linear(args.d, args.c)
    criterion = nn.CrossEntropyLoss()
if args.loss == 'tree':
    new2index, length, tree = CoverTreeLoss.tree_structure(args.c, W_proj, base=2)
    criterion = CoverTreeLoss(args.c, length, args.d, new2index)
    model = nn.Linear(args.d, length)
if args.loss == 'simloss':
    sim_matrix = gen_sim(U)
    sim_matrix = (sim_matrix - args.lower_bound) / (1 - args.lower_bound)
    sim_matrix[sim_matrix < 0.0] = 0.0
    model = nn.Linear(args.d, args.c)
    criterion = SimLoss(w=sim_matrix)
if args.loss == 'HSM':
    new2index, index2brother, length = HSM.tree_structure(args.c,U)
    model = nn.Linear(args.d, args.c)
    criterion = HSM(args.c, args.d, new2index, index2brother, length)

optimizer = optim.SGD(
    model.parameters(),
    lr=args.lr,
    momentum=args.momentum,
    weight_decay=args.weight_decay,
    )
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n)

# training
for i in range(args.n):
    # calculate the loss
    if args.loss == 'xentropy':
        W = model.weight
        logits = model(X[i].view(1,args.d))
        loss = criterion(logits, Y[i].view(1)) 
    if args.loss == 'tree':
        W_ = model.weight
        loss, logits, W = criterion(W_, X[i].view(1,args.d), Y[i].view(1))
    if args.loss == 'simloss':
        W = model.weight
        logits = model(X[i].view(1,args.d))
        prob = F.softmax(logits, dim=-1)
        loss = criterion(prob, Y[i].view(1))
    if args.loss == 'HSM':
        W = model.weight
        logits = model(X[i].view(1,args.d))
        loss = criterion(logits, Y[i].view(1))

    # backprop
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    # W_err = torch.norm(torch.abs(W - torch.Tensor(W_star))) 
    # prob = F.softmax(logits, dim=-1)
    # _, pred = torch.max(prob, dim=-1)
    # if pred == Y[i]:
    #     correct += 1
    # accuracy = correct/(i+1) 

# testing
correct = 1e-10
model.eval()
for i in range(args.test):
    # calculate the loss
    if args.loss == 'xentropy':
        W = model.weight
        logits = model(X_[i].view(1,args.d))
        loss = criterion(logits, Y_[i].view(1)) 
    if args.loss == 'tree':
        W_ = model.weight
        loss, logits, W = criterion(W_, X_[i].view(1,args.d), Y_[i].view(1))
    if args.loss == 'simloss':
        W = model.weight
        logits = model(X_[i].view(1,args.d))
        prob = F.softmax(logits, dim=-1)
        loss = criterion(prob, Y_[i].view(1))
    if args.loss == 'HSM':
        W = model.weight
        logits = model(X_[i].view(1,args.d))
        loss = criterion(logits, Y_[i].view(1))

    W_err = torch.norm(torch.abs(W - torch.Tensor(W_star))) 
    prob = F.softmax(logits, dim=-1)
    _, pred = torch.max(prob, dim=-1)
    if pred == Y_[i]:
        correct += 1

accuracy = correct/10000


print(accuracy)
# with open(f'{args.experiment}_{args.loss}_{args.lr}.txt', 'a') as f:
#     f.write(f'Loss: {loss} \t W_err: {W_err} \t Accuracy: {accuracy} \n')
