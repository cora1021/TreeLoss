import argparse
parser = argparse.ArgumentParser(description='PyTorch CiFar100 Example')
parser.add_argument('--batch_size', type=int, default=256)                      
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.15, metavar='LR')
parser.add_argument('--weight_decay', type=float, default=8e-4)
parser.add_argument('--seed', type=int, default=666)
parser.add_argument('--dimension', type=int, default=64)
args = parser.parse_args()

from cover_tree import CoverTree
from loss import CoverTreeLoss
import numpy as np
import random
import time
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from resnet import *
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/cifar100_50/covertree/lr0.15-dropout0.5')

device = torch.device("cuda")
from utilities import gen_data, gen_matrix, set_logger, set_seed, _print, path, norm
set_seed(args)

def train(model, device, data, label, train_original, m, batch_size, optimizer, criterion):
    model.train()
    pred = []
    correct = 0
    for start in range(0, len(data), batch_size):
        train_X = torch.FloatTensor(data[start:start+batch_size])
        train_Y = torch.LongTensor(label[start:start+batch_size])
        train_X, train_Y = train_X.to(device), train_Y.to(device)
        loss, logits = model(train_X, train_Y)
        pred += criterion.predict(logits)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    for i in range(len(label)):
        pred_num = pred[i]
        label_num = train_original[i]
        correct += m[pred_num, label_num]
            
    accuracy = correct/len(label)
    return accuracy, loss.item()
    
def test(model, device, data, label, original_label, m, batch_size, criterion):
    model.eval()
    test_pred = []
    correct = 0
    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            test_X = torch.FloatTensor(data[start:start+batch_size])
            test_Y = torch.LongTensor(label[start:start+batch_size])
            test_X, test_Y = test_X.to(device), test_Y.to(device)
            loss, logits = model(test_X, test_Y)
            test_pred += criterion.predict(logits)

            test_loss = loss.item()

        for i in range(len(label)):
            pred_num = test_pred[i]
            label_num = original_label[i]
            correct += m[pred_num, label_num]
            
        accuracy = correct/len(label)
        
        return accuracy, test_loss

# Data
normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])
train_dataset = torchvision.datasets.CIFAR100(
            root='./cifar100', train=True, download=True,
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,]))
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)

test_dataset = torchvision.datasets.CIFAR100(
            root='./cifar100', train=False, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,]))
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

timestamp = str(time.time())

for number in range(50, 101):
    c = 100
    m = gen_matrix(c, number)
    credit = norm(m)

    train_data, train_label, train_original, test_data, test_label, test_original, sim_matrix = gen_data(trainloader, testloader, m)
    label_list = []
    for ele in train_label+test_label:
        label_list.append(ele)
    target_list = list(set(label_list))

    new2index, length = CoverTreeLoss.tree_structure(c,m)
    criterion = CoverTreeLoss(c, length, args.dimension, new2index)
    model = resnet20_cifar(criterion, num_classes=length).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                        momentum=0.9, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    train_data = np.array(train_data)
    train_label = np.array(train_label)

    for epoch in tqdm(range(args.epochs)):
        train_dataset = list(zip(train_data, train_label, train_original))
        random.shuffle(train_dataset)
        train_data, train_label, train_original = zip(*train_dataset)

        accuracy, loss = train(model, device, train_data, train_label, train_original, credit, args.batch_size, optimizer, criterion)
        scheduler.step()
        
        test_accuracy, test_loss = test(model, device, test_data, test_label, test_original, credit, args.batch_size, criterion)
        print(test_accuracy)
        writer.add_scalar('Accuarcy: ', test_accuracy, epoch)
        writer.add_scalar('Loss: ', test_loss, epoch)
    
    # test_accuracy, test_loss = test(model, device, test_data, test_label, test_original, credit, args.batch_size, criterion)
    # writer.add_scalar('Accuarcy: ', test_accuracy, number)
    # writer.add_scalar('Loss: ', test_loss, number)
    # print(test_accuracy, test_loss, number)
    exit(0)