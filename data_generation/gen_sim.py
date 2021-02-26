import numpy as np, numpy.random
import random
from sklearn.metrics.pairwise import cosine_similarity
import torch

b = 10 
a = 5


def gen_matrix(b,a):
    m = np.zeros((b,a))
    for i in range(a):
        m[:, i] = np.random.dirichlet(np.ones(b),size=1).T.reshape(10)

    return m
m = gen_matrix(b,a)

def gen_sim(m):
    b, a = m.shape
    # create sim matrix
    new = []
    for i in range(b):
        new.append(m[i,:])
    cosine_dist = cosine_similarity(new)
    sim_matrix = torch.from_numpy(cosine_dist)
    # create new dataset
    new_label = []
    for i in range(b):
        draw = np.random.choice(range(a), 1, p=m[i,:])
        new_label.append(draw)

    return sim_matrix, new_label

sim, new_label = gen_sim(m)

