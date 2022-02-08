#!/usr/bin/python3

# this is a temporary hack to allow importing the files in the TreeLoss folder when the library is not yet installed
import sys
sys.path.append('../..')

import logging
import os
LOGLEVEL = os.environ.get('LOGLEVEL', 'WARNING').upper()
logging.basicConfig(
    level=LOGLEVEL,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    )

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
parser_data.add_argument('--c', type=int, default=100)
parser_data.add_argument('--n', type=int, default=1000)
parser_data.add_argument('--d', type=int, default=10)
parser_data.add_argument('--sigma', type=float, default=1.0)
parser_data.add_argument('--seed', type=int, default=666)
parser_data.add_argument('--base', type=float, default=1.1)
parser_data.add_argument('--batch', type=int, default=1000)
parser_data.add_argument('--epoch', type=int, default=1)



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
from torch.utils.tensorboard import SummaryWriter
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
            logits = self.fc(x)
            loss = self.criterion(logits, y)
        elif loss_fun == 'simloss':
            logits = self.fc(x)
            prob = self.softmax(logits)
            loss = self.criterion(prob, y)
        elif loss_fun == 'tree':
            weights = self.fc.weight
            loss, logits, added_weights = self.criterion(weights, x, y)
        elif loss_fun == 'HSM':
            logits = self.fc(x)
            loss = self.criterion(logits, y)
        return loss, logits

if args.loss == 'xentropy' :
    criterion = nn.CrossEntropyLoss().cuda()
    model = Mod(args.c, args.d, criterion).cuda()
elif args.loss == 'simloss':
    sim_matrix = gen_sim(U)
    sim_matrix = (sim_matrix - args.lower_bound) / (1 - args.lower_bound)
    sim_matrix[sim_matrix < 0.0] = 0.0
    criterion = SimLoss(w=sim_matrix.cuda())
    model = Mod(args.c, args.d, criterion).cuda()
elif args.loss == 'tree':
    new2index, length, tree = CoverTreeLoss.tree_structure(args.c,U, args.base)
    level_ = level(tree)
    height = -level_
    criterion = CoverTreeLoss(args.c, length, args.d, new2index)
    model = Mod(length, args.d, criterion).cuda()
elif args.loss == 'HSM':
    new2index, index2brother, length = HSM.tree_structure(args.c,U)
    criterion = HSM(args.c, args.d, new2index, index2brother, length)
    model = Mod(args.c, args.d, criterion).cuda()
else:
    raise NotImplementedError


logging.debug('create optimizer')
optimizer = optim.SGD(
    model.parameters(),
    lr=args.lr,
    momentum=args.momentum,
    weight_decay=args.weight_decay,
    )
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n)
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

for ep in range(args.epoch):
    loss_sum = 0
    correct = 1e-10
    train_pred = []
    for start in range(0, args.n, args.batch):
        train_X = torch.FloatTensor(X[start:start+args.batch].view(args.batch,args.d))
        train_Y = torch.LongTensor(Y[start:start+args.batch].view(args.batch))
        train_X, train_Y = train_X.cuda(), train_Y.cuda()
        loss, logits = model(train_X, train_Y, args.loss)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        prob = F.softmax(logits, dim=-1)
        _, pred = torch.max(prob, dim=-1)
        loss_sum += loss
        train_pred += pred.detach().cpu().tolist()

    for i in range(len(Y)):
        if train_pred[i] == Y[i]:
            correct += 1
            
    accuracy = correct/len(Y)
print(accuracy)


# Test set
Y_ = np.random.choice(range(args.c), size=[int(args.n*10)])

X_ = []
for i in range(int(args.n*10)):
    x_i_ = np.random.normal(W_star[Y_[i],:], args.sigma)
    X_.append(x_i_)
X_ = np.array(X_)

Y_ = torch.LongTensor(Y_)
X_ = torch.Tensor(X_)
model.eval()
correct = 1e-10
test_loss_sum = 0 
test_pred = []
for start in range(0, args.n*10, args.batch):
    test_X = torch.FloatTensor(X_[start:start+args.batch].view(args.batch,args.d))
    test_Y = torch.LongTensor(Y_[start:start+args.batch].view(args.batch))
    test_X, test_Y = test_X.cuda(), test_Y.cuda()
    loss, logits = model(test_X, test_Y, args.loss)

    prob = F.softmax(logits, dim=-1)
    _, pred = torch.max(prob, dim=-1)
    test_loss_sum += loss
    test_pred += pred.detach().cpu().tolist()


for i in range(len(Y_)):
    if test_pred[i] == Y_[i]:
        correct += 1
        
test_accuracy = correct/len(Y_)
print(test_accuracy)
# loss_ = loss_sum
# test_loss_ = test_loss_sum/10
# with open(f'base_experiment_{args.loss}_50.txt', 'a') as f:
#     f.write(f'Loss: {loss_} \t Accuracy: {accuracy} \t Generlization Loss: {test_loss_} \t Test Accuracy: {test_accuracy}\n')
# # # # Height: {height} \t Training Loss: {training loss} Generlization Loss: {genarlization loss}
