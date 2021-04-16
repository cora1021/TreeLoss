import numpy as np, numpy.random
import random
import os
import torch
from sklearn.metrics.pairwise import cosine_similarity

def gen_data(trainloader, testloader, m):
    """
    This function generate data with new labels.
    New labels are generated by adding a new number which selected uniformly at random from [number].
    """
    sim = gen_sim(m)
    examples = enumerate(trainloader)

    train_data = []
    train_label = []
    while True:
        try:
            batch_idx, (example_data, example_targets) = next(examples)
            label = example_targets.numpy()
            new_train = np.random.choice(np.arange(0,m.shape[0]), p = np.transpose(m[:,label]).flatten())
            train_label.append(new_train)
            train_data.append(example_data.squeeze(0).numpy())
        except:
            break

    instances = enumerate(testloader)

    test_data = []
    test_label = []
    while True:
        try:
            batch_index, (instances_data, instances_targets) = next(instances)
            label = instances_targets.numpy()
            new_test = np.random.choice(np.arange(0,m.shape[0]), p = np.transpose(m[:,label]).flatten())
            test_label.append(new_test)
            test_data.append(instances_data.squeeze(0).numpy())
        except:
            break

    return train_data, train_label, test_data, test_label, sim

def gen_matrix(num_label, num_new):
    """
    This function get distribution of new labels.
    The input are number of original labels and number of new labels we want to create.
    It returns a matrix of shape ba (where a is the number of classes in the original dataset and b is the number of new classes we're creating;
    """
    m = np.zeros((num_new, num_label))
    for i in range(num_label):
        m[:,i] = np.random.dirichlet(np.ones(num_new),size=1)
    return m


# def gen_sim(m):
#     """
#     This function generate similarity matrix.
#     Every row of input matrix m represent every new label.
#     """

#     b, a = m.shape
#     new = []
#     for i in range(b):
#         new.append(m[i,:])
#     cosine_dist = cosine_similarity(new)
#     sim_matrix = torch.from_numpy(cosine_dist)

#     return sim_matrix

def gen_sim(m):
    I = np.identity(m.shape[0])
    sim = np.maximum(I, np.dot(m, np.transpose(m)))
    sim_matrix = torch.from_numpy(sim)
    return sim_matrix

