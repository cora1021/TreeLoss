import numpy as np
import random
import os


def gen_label(number, label):
    train_label = []
    k = np.random.randint(number)
    label.append(k)
    new_label.append(label)
    return new_label

def gen_data(trainloader, testloader, number, ):

    examples = enumerate(trainloader)

    train_data = []
    train_label = []
    while True:
        try:
            batch_idx, (example_data, example_targets) = next(examples)
            label = example_targets.numpy().tolist()
            train_label = gen(number, label)
            train_data.append(example_data.squeeze(0).numpy())
        except:
            break

    instances = enumerate(testloader)

    test_data = []
    test_label = []
    while True:
        try:
            batch_index, (instances_data, instances_targets) = next(instances)
            origin_label = instances_targets.numpy().tolist()
            test_label = gen(number, origin_label)
            test_data.append(instances_data.squeeze(0).numpy())
        except:
            break

    return train_data, train_label, test_data, test_label

    label_list = []
    for ele in train_label+test_label:
        label_list.append(tuple(ele))
    target_list = list(set(label_list))
    label2index, index2label = get_labels(target_list)



def gen_sim(target_list):

    lower_bound = 0.5
    similarity_matrix = np.zeros((len(target_list), len(target_list)))

    for i in range(len(target_list)):
        for j in range(len(target_list)):
            if target_list[i][0] == target_list[j][0]:
                similarity_matrix[i][j] = 1
    sim_matrix = torch.from_numpy(similarity_matrix)
    sim_matrix = (sim_matrix - lower_bound) / (1 - lower_bound)
    sim_matrix[sim_matrix < 0.0] = 0.0

    return sim_matrix
