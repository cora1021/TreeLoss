#!/usr/bin/python3

# this is a temporary hack to allow importing the files in the TreeLoss folder when the library is not yet installed
import sys
sys.path.append('.')

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
parser_data.add_argument('--a', type=int, default=5)
parser_data.add_argument('--c', type=int, default=10)
parser_data.add_argument('--n', type=int, default=100)
parser_data.add_argument('--d', type=int, default=64)
parser_data.add_argument('--sigma', type=float, default=1.0)
parser_data.add_argument('--seed', type=int, default=666)

parser_model = parser.add_argument_group(
        title='model hyperparameters',
        description="it's recommended to always use scientific notation for hyperparameter values since we really care about the order of magnitude",
        )
parser_model.add_argument('--lr', type=float, default=1e-4, metavar='LR') 
parser_model.add_argument('--max_iter', type=int, default=int(1e1))
parser_model.add_argument('--momentum', type=float, default=0.9)
parser_model.add_argument('--weight_decay', type=float, default=3e-4)
parser_model.add_argument('--loss', choices=['tree','xentropy'], required=True)

parser_debug = parser.add_argument_group(title='debug')
parser_model.add_argument('--logdir', default='log')
args = parser.parse_args()

# load imports;
# we do this after parsing the command line arguments because it takes a long time,
# and we want immediate feedback from the command line parser if we have an invalid argument or pass the --help option
logging.info('load imports')
import random
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# set the seed
def set_seed(seed):
    logging.debug('set_seed('+str(args.seed)+')')
    random.seed(seed) 
    np.random.seed(seed) 
    torch.manual_seed(seed) 
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed_all(seed) 
set_seed(args.seed)

# FIXME:
# the code below doesn't work because I can't import cover_tree;
# add the required libraries to requirements.txt
# FIXME:
# your set_seed function should take an int as a parameter and not args; see above
'''
from TreeLoss.utilities import set_seed
set_seed(args.seed)
'''

################################################################################
# generate the data
################################################################################
logging.info('generating the data')

# generate the model hyperparameters
# NOTE: variable names match exactly the names in the paper
U = np.random.normal(size=[args.c, args.a])
V = np.random.normal(size=[args.a, args.d])
W_star = U @ V

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

Y = torch.Tensor(Y)
X = torch.Tensor(X)
logging.debug("Y.shape="+str(Y.shape))
logging.debug("X.shape="+str(X.shape))

################################################################################
# define the model
################################################################################
logging.info('defining the model')

model = nn.Linear(args.d, args.c)
criterion = nn.CrossEntropyLoss()

################################################################################
# train the model
################################################################################
logging.info('training the model')

logging.debug('create SummaryWriter')
# FIXME: uncomment
#experiment_name=f'a={args.a},c={args.c},d={args.d},n={args.n},sigma={args.sigma},lr={args.lr},loss={args.loss},seed={args.seed}'
#logging.info(f'experiment_name={experiment_name}')
experiment_name='test'
writer = SummaryWriter(os.path.join(args.logdir, experiment_name))

logging.debug('create optimizer')
optimizer = optim.SGD(
    model.parameters(),
    lr=args.lr,
    momentum=args.momentum,
    weight_decay=args.weight_decay,
    )

logging.debug('training loop')
for training_iter in range(args.max_iter):

    logging.debug('training_iter='+str(training_iter))

    # i is the current data point
    i = training_iter%args.n

    # calculate the loss
    logits = model(X[i])
    loss = criterion(logits, Y[i]) # FIXME: notation is incorrect

    # backprop
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # log to tensorboard
    W_err = 0 # FIXME: should be |W-W*|
    accuracy = 0 # FIXME
    writer.add_scalar('losses/loss', loss, i)
    writer.add_scalar('losses/W_err', W_err, i)
    writer.add_scalar('losses/accuracy', accuracy, i)

################################################################################
# save the results
################################################################################
logging.info('saving results')

# FIXME:
# save the W_err to a file
