

import typing
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .cover_tree import CoverTree
from scipy.spatial.distance import cosine
from .utilities import path, _print

class CoverTreeLoss(torch.nn.Module):
    def __init__(self, k, length, d, new2index) -> None:
        '''
        k: number of existed classes
        length: number of existed classes and pseudo classes
        d: dimension of parameter matrix W(k,d)
        new2index: a dictionary which we can get a a path of a existed class
        '''
        super().__init__()
        self.c = k
        self.linear = nn.Linear(d, length, bias=False)
        self.criterion = nn.CrossEntropyLoss()
        self.new2index = new2index

    @staticmethod
    def tree_structure(k, m):
        '''
        m: embeddings of existed classes
        This function constructs cover tree structure.
        '''
        distance = cosine
        tree = CoverTree(m, distance, leafsize=1)
        _print(tree)
        new_index = path(tree)

        # new label
        start_index = k
        index_map = dict()
        inverse_index = dict()
        for p in new_index:
            for node in p[:-1]:  
                if node not in index_map:
                    index_map[node] = start_index
                    inverse_index[start_index] = node
                    start_index+=1
        for p in new_index:
            for i in range(len(p[:-1])):
                p[i] = index_map[p[i]]

        length = max(index_map.values()) + 1
        new2index = dict()
        for i in range(len(new_index)):
            new2index[new_index[i][len(new_index[i])-1]] = new_index[i]
        return new2index, length

    def forward(self,
                weights: torch.Tensor,
                x: torch.Tensor, 
                y: torch.Tensor) -> torch.Tensor:
        '''
        x: output of the model
        y: labels
        This function computes the cover tree loss.
        '''
        
        added_weights = []
        for j in range(self.k):
            path = self.new2index[j]
            to_add_list = [weights[j, :]]
            for ele in range(len(path)-1):
                to_add_list.append(weights[path[ele], :])

            added_weights.append(torch.stack(to_add_list, dim=0).sum(dim=0))

        added_weights = torch.stack(added_weights, dim=0) 
        logits = torch.matmul(x, added_weights.transpose(0, 1)) 
        
        loss = self.criterion(logits, y)
        return loss, logits, added_weights
