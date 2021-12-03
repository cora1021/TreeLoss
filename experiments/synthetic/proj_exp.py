
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
parser_data.add_argument('--num', type=int, default=1)
parser_data.add_argument('--a', type=int, default=5)
parser_data.add_argument('--k', type=int, default=100)
parser_data.add_argument('--d_', type=int, default=5)
parser_data.add_argument('--n', type=int, default=1000)
parser_data.add_argument('--d', type=int, default=1000)
parser_data.add_argument('--sigma', type=float, default=1.0)
parser_data.add_argument('--seed', type=int, default=666)
parser_data.add_argument('--test', type=int, default=10000)
parser_data.add_argument('--batch', type=int, default=10000)

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
parser_model.add_argument('--experiment', choices=['loss_vs_n','loss_vs_d', 'loss_vs_sigma', 'loss_vs_c', 'loss_vs_d_'], required=True)
args = parser.parse_args()

import random
import numpy as np
from numpy.linalg import inv
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

U = np.random.normal(size=[args.k, args.a])
V = np.random.normal(size=[args.a, args.d])
W_star = U @ V

# generating projection matrix
A = np.random.normal(size=[args.d, args.d_])
u, s, vh = np.linalg.svd(A, full_matrices=True)
proj_matrix = u*u.T
R = proj_matrix[:, :args.d_]

W_proj = W_star @ R

# training set
Y = np.random.choice(range(args.k), size=[args.n])
X = []
for i in range(args.n):
    x_i = np.random.normal(W_star[Y[i],:], args.sigma)
    X.append(x_i)
X = np.array(X)

Y = torch.LongTensor(Y)
X = torch.Tensor(X)

# testing set
Y_ = np.random.choice(range(args.k), size=[args.test])
X_ = []
for i in range(args.test):
    x_i = np.random.normal(W_star[Y_[i],:], args.sigma)
    X_.append(x_i)
X_ = np.array(X_)

Y_ = torch.LongTensor(Y_)
X_ = torch.Tensor(X_)

class Mod(torch.nn.Module):
    def __init__(self, class_num, dimension, criterion, epsilon: float = 1e-8) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.fc = nn.Linear(dimension, class_num, bias=False)
        self.criterion = criterion

    def forward(self,
                x: torch.Tensor, #(batch_size, hidden_size)
                y: torch.Tensor, loss_fun) -> torch.Tensor:
    
        if loss_fun == 'xentropy':
            weights = self.fc.weight
            logits = self.fc(x)
            loss = self.criterion(logits, y)
        elif loss_fun == 'simloss':
            weights = self.fc.weight
            logits = self.fc(x)
            prob = self.softmax(logits)
            loss = self.criterion(prob, y)
        elif loss_fun == 'tree':
            weights_ = self.fc.weight
            loss, logits, weights = self.criterion(weights_, x, y)
        elif loss_fun == 'HSM':
            weights = self.fc.weight
            logits = self.fc(x)
            loss = self.criterion(logits, y)
        return loss, logits

if args.loss == 'xentropy' :
    criterion = nn.CrossEntropyLoss().cuda()
    model = Mod(args.k, args.d, criterion).cuda()
elif args.loss == 'simloss':
    sim_matrix = gen_sim(U)
    sim_matrix = (sim_matrix - args.lower_bound) / (1 - args.lower_bound)
    sim_matrix[sim_matrix < 0.0] = 0.0
    criterion = SimLoss(w=sim_matrix.cuda())
    model = Mod(args.k, args.d, criterion).cuda()
elif args.loss == 'tree':
    new2index, length = CoverTreeLoss.tree_structure(args.k, W_proj)
    criterion = CoverTreeLoss(args.k, length, args.d, new2index)
    model = Mod(length, args.d, criterion).cuda()
elif args.loss == 'HSM':
    new2index, index2brother, length = HSM.tree_structure(args.k,U)
    criterion = HSM(args.k, args.d, new2index, index2brother, length)
    model = Mod(args.k, args.d, criterion).cuda()
else:
    raise NotImplementedError

optimizer = optim.SGD(
    model.parameters(),
    lr=args.lr,
    momentum=args.momentum,
    weight_decay=args.weight_decay,
    )
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n)

# training
model.train()
train_pred = []
for start in range(0, args.n, 1):
    train_X = torch.FloatTensor(X[start:start+1].view(1,args.d))
    train_Y = torch.LongTensor(Y[start:start+1].view(1))
    train_X, train_Y = train_X.cuda(), train_Y.cuda()
    loss, logits = model(train_X, train_Y, args.loss)
    prob = F.softmax(logits, dim=-1)
    _, pred = torch.max(prob, dim=-1)
    train_pred += pred.detach().cpu().tolist()

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
correct = 0
for i in range(len(Y)):
    if train_pred[i] == Y[i]:
        correct += 1
accuracy = correct/len(Y)

# testing

correct = 1e-10
test_pred = []
model.eval()
for start in range(0, args.test, args.batch):
    test_X = torch.FloatTensor(X_[start:start+args.batch].view(args.batch,args.d))
    test_Y = torch.LongTensor(Y_[start:start+args.batch].view(args.batch))
    test_X, test_Y = test_X.cuda(), test_Y.cuda()
    loss, logits = model(test_X, test_Y, args.loss)

    prob = F.softmax(logits, dim=-1)
    _, pred = torch.max(prob, dim=-1)
    # W_err = torch.norm(torch.abs(W - torch.Tensor(W_star))) 
    test_pred += pred.detach().cpu().tolist()

for i in range(len(Y_)):
    if test_pred[i] == Y_[i]:
        correct += 1
        
accuracy = correct/len(Y_)
print(accuracy)
# with open(f'U_{args.loss}_{args.k}_{args.n}.txt', 'a') as f:
#     f.write(f'Accuracy: {accuracy} \n')
