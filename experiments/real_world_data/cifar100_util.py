import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from pathlib import Path
from pathlib import Path
import click
import numpy as np
from tqdm import tqdm
import sys
import time
import logging
import numpy as np, numpy.random
import random
import os
import torch
from sklearn.metrics.pairwise import cosine_similarity
import math

def filter_data(trainloader, testloader, class_black):
    examples = enumerate(trainloader)

    train_data = []
    train_class = []
    while True:
        try:
            batch_idx, (example_data, example_targets) = next(examples)
            if example_targets not in class_black:
                train_class.append(example_targets)
                train_data.append(example_data.squeeze(0).numpy())
        except:
            break
    train_dataset = list(zip(train_data, train_class))
    random.shuffle(train_dataset)
    train_data, train_class = zip(*train_dataset)
    train_data = np.array(train_data)
    train_class = np.array(train_class)

    instances = enumerate(testloader)

    test_data = []
    test_class = []
    while True:
        try:
            batch_index, (instances_data, instances_targets) = next(instances)
            if instances_targets not in class_black:
                test_class.append(instances_targets)
                test_data.append(instances_data.squeeze(0).numpy())
        except:
            break
    test_data = np.array(test_data)
    test_class = np.array(test_class)
    return train_data, train_class, test_data, test_class

def encode(class_black, train_label, test_label):
    label2class = dict()
    label_original = list(range(0,100))
    for c in class_black:
        label_original.remove(c)
    for i in label_original:
        label2class[i] = label_original.index(i)

    train_class = []
    test_class = []
    for i in train_label:
        train_class.append(label2class[i])
    for i in test_label:
        test_class.append(label2class[i])

    train_class = np.array(train_class)
    test_class = np.array(test_class)
    return label2class, train_class, test_class



def train(model, device, data, label, batch_size, optimizer, loss_fun):
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
        if train_pred[i] == label[i]:
            correct += 1
            
    accuracy = correct/len(label)
    return accuracy, loss.item()
    
def test(model, device, data, label, batch_size, loss_fun, sim_matrix):
    model.eval()
    test_pred = []
    correct = 0
    sim_correct = 0
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
            if test_pred[i] == label[i]:
                correct += 1
            sim_correct += sim_matrix[test_pred[i], label[i]]
            
            
        accuracy = correct/len(label)
        sim_accuracy = sim_correct/len(label)
        return accuracy, sim_accuracy, test_loss, test_pred

class LabelField:
    def __init__(self):
        self.label2id = dict()
        self.label_num = 0

    def get_id(self, label):
        
        if label in self.label2id:
            return self.label2id[label]
        
        self.label2id[label] = self.label_num
        self.label_num += 1

        return self.label2id[label]

    def contains(self, label):

        return label in self.label2id

    def __len__(self):

        return len(self.label2id)

