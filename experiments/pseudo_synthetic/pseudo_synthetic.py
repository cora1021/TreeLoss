import sys
sys.path.append('../..')

import argparse
parser = argparse.ArgumentParser(description='Pseudo Synthetic Data Experiment')
parser.add_argument('--data', choices=['mnist', 'cifar10', 'cifar100'], required=True)
parser.add_argument('--loss', choices=['xentropy', 'simloss', 'covertreeloss'], required=True)
parser.add_argument('--seed', type=int, default=666)

parser.add_argument('--batch_size', type=int, default=256)                      
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.15, metavar='LR')
parser.add_argument('--weight_decay', type=float, default=8e-4)
parser.add_argument('--d', type=int, default=64)
parser.add_argument('--lower_bound', type=float, default=0.5)
args = parser.parse_args()

import numpy as np
import random
import time
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from resnet import *
from data_loader import data_loader
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(f'runs/{args.data}/{args.loss}')

device = torch.device("cuda")
from TreeLoss.loss import CoverTreeLoss, SimLoss
from TreeLoss.utilities import gen_data, gen_matrix, set_logger, set_seed, _print, path, norm
set_seed(args.seed)

################################################################################
# training/testing functions
################################################################################
def train(model, device, data, label, train_original, m, batch_size, optimizer, loss_fun):
    model.train()
    train_pred = []
    correct = 0
    for start in range(0, len(data), batch_size):
        train_X = torch.FloatTensor(data[start:start+batch_size])
        train_Y = torch.LongTensor(label[start:start+batch_size])
        train_X, train_Y = train_X.to(device), train_Y.to(device)
        loss, logits = model(train_X, train_Y, loss_fun)
        prob = F.softmax(logits, dim=-1)

        _, pred = torch.max(prob, dim=-1)
        train_pred += pred.detach().cpu().tolist()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    for i in range(len(label)):
        pred_num = train_pred[i]
        label_num = train_original[i]
        correct += m[pred_num, label_num]
            
    accuracy = correct/len(label)
    return accuracy, loss.item()
    
def test(model, device, data, label, original_label, m, batch_size, loss_fun):
    model.eval()
    test_pred = []
    correct = 0
    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            test_X = torch.FloatTensor(data[start:start+batch_size])
            test_Y = torch.LongTensor(label[start:start+batch_size])
            test_X, test_Y = test_X.to(device), test_Y.to(device)
            loss, logits = model(test_X, test_Y, loss_fun)
            prob = F.softmax(logits, dim=-1)
            _, pred = torch.max(prob,dim=-1)

            test_loss = loss.item()
            test_pred+=pred.detach().cpu().tolist()

        for i in range(len(label)):
            pred_num = test_pred[i]
            label_num = original_label[i]
            correct += m[pred_num, label_num]
            
        accuracy = correct/len(label)
        
        return accuracy, test_loss

################################################################################
# load the data
################################################################################
trainloader, testloader = data_loader(args.data)
timestamp = str(time.time())
if args.data == 'mnist' or 'cifar10':
    c = 10
if args.data == 'cifar100':
    c = 100

################################################################################
# experiment
################################################################################
for number in range(1, c):
    # define data
    m = gen_matrix(c, number)
    credit = norm(m)
    train_data, train_label, train_original, test_data, test_label, test_original, sim_matrix = gen_data(trainloader, testloader, m)

    # define the model
    if args.loss == 'xentropy' :
        criterion = nn.CrossEntropyLoss().cuda()
        model = resnet20_cifar(criterion, c).to(device)
    elif args.loss == 'simloss':
        sim_matrix = (sim_matrix - args.lower_bound) / (1 - args.lower_bound)
        sim_matrix[sim_matrix < 0.0] = 0.0
        criterion = SimLoss(w=sim_matrix.cuda())
        model = resnet20_cifar(criterion, c).to(device)
    elif args.loss == 'covertreeloss':
        new2index, length = CoverTreeLoss.tree_structure(c,m)
        criterion = CoverTreeLoss(c, length, args.d, new2index)
        model = resnet20_cifar(criterion, length).to(device)
    else:
        raise NotImplementedError

    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                        momentum=0.9, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    train_data = np.array(train_data)
    train_label = np.array(train_label)

    # train the model
    for epoch in tqdm(range(args.epochs)):
        train_dataset = list(zip(train_data, train_label, train_original))
        random.shuffle(train_dataset)
        train_data, train_label, train_original = zip(*train_dataset)

        accuracy, loss = train(model, device, train_data, train_label, train_original, credit, args.batch_size, optimizer, args.loss)
        scheduler.step()
        
        test_accuracy, test_loss = test(model, device, test_data, test_label, test_original, credit, args.batch_size, args.loss)
        print(test_accuracy)
        writer.add_scalar('Accuarcy: ', test_accuracy, epoch)
        writer.add_scalar('Loss: ', test_loss, epoch)
    
    # test_accuracy, test_loss = test(model, device, test_data, test_label, test_original, credit, args.batch_size, args.loss)
    # writer.add_scalar('Accuarcy: ', test_accuracy, number)
    # writer.add_scalar('Loss: ', test_loss, number)
    # print(test_accuracy, test_loss, number)
    