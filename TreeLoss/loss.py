import typing
import torch
from torch._C import FloatStorageBase
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class SimLoss(torch.nn.Module):
    def __init__(self,
                 w: typing.Optional[torch.Tensor],
                 lower_bound: float = 0.5,
                 epsilon: float = 1e-8) -> None:
        super().__init__()

        assert lower_bound >= 0.0
        assert lower_bound < 1.0

        self.w = w.float()
        self.epsilon = epsilon

    def forward(self,
                x: torch.Tensor,
                y: torch.Tensor) -> torch.Tensor:
        w = self.w[y, :]
        print(w.size())
        print(x.size())
        return torch.mean(-torch.log(torch.sum(w * x, dim=1) + self.epsilon))

    # def __repr__(self) -> str:
    #     return "SimCE"

class CoverTreeLoss(torch.nn.Module):
    def __init__(self, true_class_num, class_num, new2index, hidden_size, epsilon: float = 1e-8) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.true_class_num = true_class_num
        self.new2index = new2index
        self.linear = nn.Linear(hidden_size, class_num, bias=False)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self,
                x: torch.Tensor, #(batch_size, hidden_size)
                y: torch.Tensor) -> torch.Tensor:

        weights = self.linear.weight #(class_num, hidden_size)
        
        added_weights = []
        for j in range(self.true_class_num):
            path = self.new2index[j]
            to_add_list = [weights[j, :]]
            for ele in range(len(path)-1):
                to_add_list.append(weights[path[ele], :])

            added_weights.append(torch.stack(to_add_list, dim=0).sum(dim=0))

        added_weights = torch.stack(added_weights, dim=0) # (true_class_num, hidden_size)
        logits = torch.matmul(x, added_weights.transpose(0, 1)) # (batch_size, true_class_num)
        
        loss = self.criterion(logits, y)
        return loss, logits

    def predict(self, x):

        prob = F.softmax(x, dim=-1)

        _, pred = torch.max(prob, dim=-1)
        _pred = pred.detach().cpu().tolist()

        return _pred
        




