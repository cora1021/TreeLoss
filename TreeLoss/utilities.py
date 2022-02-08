import sys
import time
import logging
import numpy as np
import random
import os
import torch
from sklearn.metrics.pairwise import cosine_similarity
from .cover_tree import CoverTree
from tqdm import tqdm
import math

def set_logger(name, timestamp):
    formatter=logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    logger=logging.getLogger(f'{name}-Logger')
    file_handler=logging.FileHandler(f'./log-{name}-{timestamp}.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler=logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    return logger

def set_seed(seed):
    random.seed(seed) 
    np.random.seed(seed) 
    torch.manual_seed(seed) 
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed_all(seed) 

def gen_sim(m):
    """
    This function generate similarity matrix.
    Every row of input matrix m represent every new label.
    """

    b, a = m.shape
    new = []
    for i in range(b):
        new.append(m[i,:])
    cosine_dist = cosine_similarity(new)
    sim_matrix = torch.from_numpy(cosine_dist)

    return sim_matrix

def get_labels(label):
  
  label2index = dict()
  index2label = dict()
  for idx, element in enumerate(label):
    label2index[element] = idx
    index2label[idx] = element

  return label2index, index2label

def _print(tree):
    def print_node(node, indent):
        if isinstance(node, CoverTree._LeafNode):
            print ("-" * indent, node)
        else:
            print ("-" * indent, node)
            for child in node.children:
                print_node(child, indent + 1)
    print_node(tree.root, 0)

def path(tree, return_idx = True):

    pathes = []
    path_tmp = []

    def _path(node, path_tmp):
        
        if isinstance(node, CoverTree._LeafNode):
            if return_idx:
                for number in node.idx:
                    path_tmp.append(number)
                    pathes.append(list(path_tmp))
                    path_tmp.pop()
            else:
                path_tmp.append(node)
                pathes.append(list(path_tmp))
                path_tmp.pop()
        else:
            if return_idx:
                path_tmp.append(node.ctr_idx)
            else:
                path_tmp.append(node)
            for child in node.children:
                _path(child, path_tmp)
            path_tmp.pop()

    _path(tree.root, path_tmp)

    return pathes

def level(tree):
    levels = []
    def _level(node):
        if isinstance(node, CoverTree._InnerNode):
            levels.append(node.level)
            for child in node.children:
                _level(child)

    _level(tree.root)

    return min(levels)



def get_brothernode(tree):

    pathes = path(tree, return_idx = False)
    results = []
    for i in pathes:
        path_bro = []
        for node in i:
            if not isinstance(node, CoverTree._LeafNode):
                path_bro.append([child.ctr_idx for child in node.children])

        results.append(path_bro)

    return results



def norm(x):
    m = np.max(x)
    n = np.min(x)
    x = (x-n) / (m-n)
    return x

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def adjust_learning_rate(optimizer, epoch, init_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = init_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

