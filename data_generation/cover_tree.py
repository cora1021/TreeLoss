
import math
import torch

class newNode: 
    def __init__(self, data): 
        self.data = data  
        self.left = self.right = None

def gen_node(k):
    num = math.ceil(math.log2(k) + 1)
    n = 2**(num-1) - 1 + k
    data = list(range(n))
    return data

def insertLevelOrder(data, root, i, n): 

    if i < n: 
        temp = newNode(data[i])  
        root = temp  

        root.left = insertLevelOrder(data, root.left, 
                                     2 * i + 1, n)  
  
        root.right = insertLevelOrder(data, root.right, 
                                      2 * i + 2, n) 
    return root 


def hasPath(root, arr, x): 

    if (not root): 
        return False
 
    arr.append(root.data)      
      
    if (root.data == x):      
        return True

    if (hasPath(root.left, arr, x) or 
        hasPath(root.right, arr, x)):  
        return True
  
    arr.pop(-1)  
    return False

def Path(root, x):  
    arr = []  

    if (hasPath(root, arr, x)): 
        return arr      
    else: 
        return None

def get_vector(data, len_vector, device):

    vec = dict()
    for i in range(len(data)):
        new_vec = torch.nn.Parameter(torch.zeros(len_vector))
        torch.nn.init.normal_(new_vec, mean=0, std=1.0)
        vec[data[i]] = new_vec.to(device)
    return vec

class CovertreeLoss(torch.nn.Module):
    def __init__(self, num_label, device):
        super().__init__()
        data = gen_node(num_label)
        label = data[-num_label:]
        n = len(data)
        root = None
        root = insertLevelOrder(data, root, 0, n)
        path = []  
        for i in label:
            path.append(Path(root, i))
        len_vector = num_label
        vec = get_vector(data[1:], len_vector, device)

        path_vector = []

        for line in path:
            path_vector.append([])
            for node in line:
                if node != root.data:
                    path_vector[-1].append(vec[node])
        
        self.label_vector = []

        for line in path_vector:
            self.label_vector.append(sum(line))


        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        
        _labels = labels.detach().cpu().tolist()
        for i in range(len(_labels)):
            logits[i,:] += self.label_vector[_labels[i]]

        return self.loss(logits, labels)

    
