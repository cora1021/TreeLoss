#!/usr/bin/python3

# this is a temporary hack to allow importing the files in the TreeLoss folder when the library is not yet installed
import sys
sys.path.append('../..')

# setup logging using an environment variable;
# by calling python with an invocation like
# $ LOGLEVEL=WARNING python3 ...
# $ LOGLEVEL=INFO python3 ...
# $ LOGLEVEL=DEBUG python3 ...
# we can set how much logging info will be printed
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
parser_data.add_argument('--d', type=int, default=64)
parser_data.add_argument('--sigma', type=float, default=1.0)
parser_data.add_argument('--random', type=float, default=0.0)
parser_data.add_argument('--seed', type=int, default=666)


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
parser_model.add_argument('--experiment', choices=['loss_vs_n','loss_vs_d', 'loss_vs_sigma', 'loss_vs_c', 'loss_vs_structure'], required=True)
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
from TreeLoss.utilities import set_seed, gen_sim
from TreeLoss.loss import CoverTreeLoss, SimLoss, HSM

# set the seed
logging.debug('set_seed('+str(args.seed)+')')
set_seed(args.exp_num*10)

# FIXME:
# the code below doesn't work because I can't import cover_tree;
# add the required libraries to requirements.txt
# FIXME:
# your set_seed function should take an int as a parameter and not args; see above

################################################################################
# generate the data
################################################################################
logging.info('generating the data')

# generate the model hyperparameters
# NOTE: variable names match exactly the names in the paper
U = np.random.normal(size=[args.c, args.a])
V = np.random.normal(size=[args.a, args.d])
W_star = U @ V

factor = np.random.normal(0,args.random)
# U_ = U + factor
U_random = np.random.normal(size=[args.c, args.a])
U_ = (1-args.random)*U + args.random*U_random
# U_ = np.random.normal(100.0, 1.0, size=[args.c, args.a])

logging.debug("U.shape="+str(U.shape))
logging.debug("V.shape="+str(V.shape))
logging.debug("W_star.shape="+str(W_star.shape))

# generate the data
Y = np.random.choice(range(args.c), size=[args.n])

# NOTE:
# it's possible to generate the X below in a single numpy command;
# that would be faster, but a bit less obvious what's going on;
# if this code is a bottleneck, however, it'd be worth doing
X = []
for i in range(args.n):
    x_i = np.random.normal(W_star[Y[i],:], args.sigma)
    X.append(x_i)
X = np.array(X)

Y = torch.LongTensor(Y)
X = torch.Tensor(X)
logging.debug("Y.shape="+str(Y.shape))
logging.debug("X.shape="+str(X.shape))

################################################################################
# define the model
################################################################################
logging.info('defining the model')
if args.loss == 'xentropy':
    model = nn.Linear(args.d, args.c)
    criterion = nn.CrossEntropyLoss()
if args.loss == 'tree':
    new2index, length, tree = CoverTreeLoss.tree_structure(args.c,U_, base=2)
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
################################################################################
# train the model
################################################################################
logging.info('training the model')

logging.debug('create SummaryWriter')
# FIXME: uncomment
experiment_name=f'a={args.a},c={args.c},d={args.d},n={args.n},sigma={args.sigma},lr={args.lr},loss={args.loss},seed={args.seed}'
logging.info(f'experiment_name={experiment_name}')
# writer = SummaryWriter(os.path.join(args.logdir, experiment_name))

logging.debug('create optimizer')
optimizer = optim.SGD(
    model.parameters(),
    lr=args.lr,
    momentum=args.momentum,
    weight_decay=args.weight_decay,
    )
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n)
logging.debug('training loop')

correct = 1e-10
for i in range(args.n):
    # calculate the loss
    if args.loss == 'xentropy':
        W = model.weight
        logits = model(X[i].view(1,args.d))
        loss = criterion(logits, Y[i].view(1)) # FIXME: notation is incorrect
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
    # log to tensorboard
    W_err = torch.norm(torch.abs(W - torch.Tensor(W_star))) # FIXME: should be |W-W*|
    prob = F.softmax(logits, dim=-1)
    _, pred = torch.max(prob, dim=-1)
    if pred == Y[i]:
        correct += 1
    accuracy = correct/(i+1) # FIXME

    # writer.add_scalar('losses/loss', loss, i)
    # writer.add_scalar('losses/W_err', W_err, i)
    # writer.add_scalar('losses/accuracy', accuracy, i)

################################################################################
# save the results
################################################################################
logging.info('saving results')

# # FIXME:
# # save the W_err to a file
loss = loss+1e-10
# print(accuracy)
with open(f'{args.experiment}_{args.loss}.txt', 'a') as f:
    f.write(f'Loss: {loss} \t W_err: {W_err} \t Accuracy: {accuracy} \n')
