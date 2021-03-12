from cover_tree import CoverTree
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/covertree')

from scipy.spatial.distance import euclidean, cityblock, chebyshev,cosine
import os
import argparse
import numpy as np
import random

import argparse
parser = argparse.ArgumentParser(description='PyTorch CiFar10 Example')
parser.add_argument('--batch_size', type=int, default=256)                      
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.1, metavar='LR')
args = parser.parse_args()

device = torch.device("cuda")

def get_labels(label):
  
  label2index = dict()
  index2label = dict()
  for idx, element in enumerate(label):
    label2index[element] = idx
    index2label[idx] = element

  return label2index, index2label

def _print(self):
    new_label = []
    def print_node(node, indent):
        if isinstance(node, CoverTree._LeafNode):
            print ("-" * indent, node)
            new_label.append(node.idx)
        else:
            print ("-" * indent, node)
            for child in node.children:
                print_node(child, indent + 1)

    print_node(self.root, 0)
    return new_label
# DLA model
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=1, padding=(kernel_size - 1) // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, xs):
        x = torch.cat(xs, 1)
        out = F.relu(self.bn(self.conv(x)))
        return out


class Tree(nn.Module):
    def __init__(self, block, in_channels, out_channels, level=1, stride=1):
        super(Tree, self).__init__()
        self.level = level
        if level == 1:
            self.root = Root(2*out_channels, out_channels)
            self.left_node = block(in_channels, out_channels, stride=stride)
            self.right_node = block(out_channels, out_channels, stride=1)
        else:
            self.root = Root((level+2)*out_channels, out_channels)
            for i in reversed(range(1, level)):
                subtree = Tree(block, in_channels, out_channels,
                               level=i, stride=stride)
                self.__setattr__('level_%d' % i, subtree)
            self.prev_root = block(in_channels, out_channels, stride=stride)
            self.left_node = block(out_channels, out_channels, stride=1)
            self.right_node = block(out_channels, out_channels, stride=1)

    def forward(self, x):
        xs = [self.prev_root(x)] if self.level > 1 else []
        for i in reversed(range(1, self.level)):
            level_i = self.__getattr__('level_%d' % i)
            x = level_i(x)
            xs.append(x)
        x = self.left_node(x)
        xs.append(x)
        x = self.right_node(x)
        xs.append(x)
        out = self.root(xs)
        return out


class DLA(nn.Module):
    def __init__(self, block=BasicBlock):
        super(DLA, self).__init__()
        self.base = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )

        self.layer1 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )

        self.layer3 = Tree(block,  32,  64, level=1, stride=1)
        self.layer4 = Tree(block,  64, 128, level=2, stride=2)
        self.layer5 = Tree(block, 128, 256, level=2, stride=2)
        self.layer6 = Tree(block, 256, 512, level=1, stride=2)
        self.linear = nn.Linear(512, len(target_list))
        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, x, y):
        out = self.base(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        logits = self.linear(out)
        prob = F.softmax(logits, dim=-1)
        # CCE
        loss = self.loss_function(logits, y)

        _, pred = torch.max(prob,dim=-1)
        return loss, pred

def train(model, device, data, label,batch_size, optimizer):
    model.train()
    pred_list = []
    for start in range(0, len(data), batch_size):
        train_X = torch.FloatTensor(data[start:start+batch_size])
        train_Y = torch.LongTensor(label[start:start+batch_size])
        train_X, train_Y = train_X.to(device), train_Y.to(device)
        loss, pred = model(train_X, train_Y)
        pred_list+=pred.detach().cpu().tolist()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return loss
    
def test(model, device, data, label, batch_size):
    model.eval()
    test_pred = []
    correct = 0
    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            test_X = torch.FloatTensor(data[start:start+batch_size])
            test_Y = torch.LongTensor(label[start:start+batch_size])
            test_X, test_Y = test_X.to(device), test_Y.to(device)
            loss, pred = model(test_X, test_Y)
            test_loss = loss.item()
            test_pred+=pred.detach().cpu().tolist()

        for i in range(len(label)):
            if test_pred[i] == label[i]:
                correct += 1
        accuracy = correct/len(label)
        return accuracy, test_loss

# Data
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./cifar10', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=1, shuffle=False)

testset = torchvision.datasets.CIFAR10(
    root='./cifar10', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=1, shuffle=False)

from gen_data import gen_data, gen_sim, gen_matrix

for number in range(1,100):

    train_data, train_label, test_data, test_label = gen_data(trainloader, testloader, number)
        
    label_list = []
    for ele in train_label+test_label:
        label_list.append(tuple(ele))

    target_list = list(set(label_list))
    label2index, index2label = get_labels(target_list)

    m = gen_matrix(label_list, label2index)
    # construct cover tree from m (distribution matrix of new labels)
    distance = cosine
    result = CoverTree(m, distance,leafsize=number)
    
    new_index = _print(result)

    new2index = dict()
    index2new = dict()
    for idx, element in enumerate(new_index):
        for i in range(len(element)):
            new2index[element[i]] = idx
        index2new[idx] = element

    train_target = []
    for ele in train_label:
        new = label2index[tuple(ele)]
        train_target.append(new2index[new])
    test_target = []
    for ele in test_label:
        new = label2index[tuple(ele)]
        test_target.append(new2index[new])

    trainset = list(zip(train_data, train_target))
    random.shuffle(trainset)
    train_data, train_target = zip(*trainset)

    model = DLA().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                        momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    for epoch in range(args.epochs):
        train_loss = train(model, device, train_data, train_target, args.batch_size, optimizer)
        scheduler.step()
        
    test_accuracy, test_loss = test(model, device, test_data, test_target, args.batch_size)
    writer.add_scalar('Accuarcy: ', test_accuracy, number)
    writer.add_scalar('Loss: ', test_loss, number)

    print(test_accuracy, test_loss, number)