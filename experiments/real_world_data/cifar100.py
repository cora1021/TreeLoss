import argparse
parser = argparse.ArgumentParser(description='Pseudo Synthetic Data Experiment')
parser.add_argument('--data', choices=['mnist', 'cifar10', 'cifar100'], required=True)
parser.add_argument('--loss', choices=['xentropy', 'simloss', 'covertreeloss', 'HSM'], required=True)
parser.add_argument('--seed', type=int, default=666)

parser.add_argument('--batch_size', type=int, default=1024)                      
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.15, metavar='LR')
parser.add_argument('--weight_decay', type=float, default=8e-4)
parser.add_argument('--d', type=int, default=64)
parser.add_argument('--lower_bound', type=float, default=0.5)
args = parser.parse_args()

from pathlib import Path
import logging
from itertools import combinations_with_replacement
from pathlib import Path
import numpy as np
from tqdm import tqdm
import gensim.downloader as api
import numpy as np
import random
import time
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from resnet import *
from TreeLoss.utilities import set_seed, gen_sim
from TreeLoss.loss import CoverTreeLoss
from data_loader import data_loader
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(f'logging/{args.data}/{args.loss}/')
device = torch.device("cuda")
from cifar100_util import *
set_seed(args.seed)

CIFAR100_SUPERCLASSES_INV = {
        'aquatic mammals': ['beaver', 'dolphin', 'otter', 'seal', 'whale'],
        'fish': ['aquarium fish', 'flatfish', 'ray', 'shark', 'trout'],
        'flowers': ['orchids', 'poppies', 'roses', 'sunflowers', 'tulips'],
        'food containers': ['bottles', 'bowls', 'cans', 'cups', 'plates'],
        'fruit and vegetables': ['apples', 'mushrooms', 'oranges', 'pears', 'sweet peppers'],
        'household electrical devices': ['clock', 'computer keyboard', 'lamp', 'telephone', 'television'],
        'household furniture': ['bed', 'chair', 'couch', 'table', 'wardrobe'],
        'insects': ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
        'large carnivores': ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
        'large man-made outdoor things': ['bridge', 'castle', 'house', 'road', 'skyscraper'],
        'large natural outdoor scenes': ['cloud', 'forest', 'mountain', 'plain', 'sea'],
        'large omnivores and herbivores': ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
        'medium-sized mammals': ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
        'non-insect invertebrates': ['crab', 'lobster', 'snail', 'spider', 'worm'],
        'people': ['baby', 'boy', 'girl', 'man', 'woman'],
        'reptiles': ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
        'small mammals': ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
        'trees': ['maple', 'oak', 'palm', 'pine', 'willow'],
        'vehicles 1': ['bicycle', 'bus', 'motorcycle', 'pickup truck', 'train'],
        'vehicles 2': ['lawnmower', 'rocket', 'streetcar', 'tank', 'tractor']
    }

    # Note: We removed classes which are not in the word embedding vocabulary
CIFAR100_BLACKLIST = ['aquarium fish', 'sweet peppers', 'computer keyboard', 'pickup truck']

CIFAR100_SUPERCLASSES = {c: sc for sc in CIFAR100_SUPERCLASSES_INV for c in CIFAR100_SUPERCLASSES_INV[sc]}
CIFAR100_CLASSES = sorted(CIFAR100_SUPERCLASSES.keys())
CIFAR100_CLASSES_FILTERED = [c for c in CIFAR100_CLASSES if c not in CIFAR100_BLACKLIST]

model = api.load('word2vec-google-news-300')

classes = [c for c in CIFAR100_CLASSES if c not in CIFAR100_BLACKLIST]

word_count = len(classes)
vector_representation = np.zeros((word_count, 300))
sim_matrix = np.zeros((word_count, word_count))
similarities = {frozenset([w1, w2]): model.similarity(w1, w2)
                    for w1, w2 in combinations_with_replacement(classes, 2)}

for i1, w1 in tqdm(enumerate(classes)):
    for i2, w2 in tqdm(enumerate(classes)):
        sim_matrix[i1, i2] = similarities[frozenset([w1, w2])]
for i in range(len(classes)):
    vector_representation[i,:] = model.wv[classes[i]]
sim_matrix = torch.from_numpy(sim_matrix)
similarity = torch.clone(sim_matrix).numpy()  

################################################################################
# experiment
################################################################################
class_black = [CIFAR100_CLASSES.index(c) for c in CIFAR100_BLACKLIST]
trainloader, testloader = data_loader(args.data)
train_data, train_label, eval_data, eval_label = filter_data(trainloader, testloader, class_black)
label2class, train_class, eval_class = encode(class_black, train_label, eval_label)

dev_data = eval_data[:int(0.2*eval_data.shape[0]),:]
dev_class = eval_class[:int(0.2*eval_class.shape[0])]

test_data = eval_data[int(0.2*eval_data.shape[0]):,:]
test_class = eval_class[int(0.2*eval_class.shape[0]):]

c = word_count
m = vector_representation
# define the model
if args.loss == 'xentropy' :
    criterion = nn.CrossEntropyLoss().cuda()
    model = resnet20_cifar(criterion, c).to(device)
elif args.loss == 'simloss':
    sim_matrix = (sim_matrix - args.lower_bound) / (1 - args.lower_bound)
    sim_matrix[sim_matrix < 0.0] = 0.0
    criterion = SimLoss(w=sim_matrix.cuda())
    model = resnet20_cifar(criterion, c).to(device)
elif args.loss == 'covertreeloss':
    new2index, length = CoverTreeLoss.tree_structure(c,m)
    criterion = CoverTreeLoss(c, length, args.d, new2index)
    model = resnet20_cifar(criterion, length).to(device)
elif args.loss == 'HSM':
    new2index, index2brother, length = HSM.tree_structure(c,m)
    criterion  = HSM(c, args.d, new2index, index2brother, length)
    model = resnet20_cifar(criterion, c).to(device)
else:
    raise NotImplementedError


optimizer = optim.SGD(model.parameters(), lr=args.lr,
                    momentum=0.9, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150, 200, 250], last_epoch=-1)

# train the model
path = f'./cifar100_model_{args.loss}.zip'
accuracy_best = 0
for epoch in tqdm(range(args.epochs)):
    train_dataset = list(zip(train_data, train_class))
    random.shuffle(train_dataset)
    train_data, train_class = zip(*train_dataset)

    accuracy, loss = train(model, device, train_data, train_class, args.batch_size, optimizer, args.loss)
    scheduler.step()
    dev_accuracy, sim_accuracy, dev_loss = test(model, device, dev_data, dev_class, args.batch_size, args.loss, similarity)
    print(dev_accuracy, dev_loss)
    writer.add_scalar('Training Accuarcy: ', accuracy, epoch)
    writer.add_scalar('Testing Accuarcy: ', dev_accuracy, epoch)
    writer.add_scalar('Training Loss: ', loss, epoch)
    writer.add_scalar('Testing Loss: ', dev_loss, epoch)
    if dev_accuracy > accuracy_best:
        accuracy_best = dev_accuracy
        torch.save({
            'model_state_dict': model.state_dict(),
            }, path)
    
    # with open(f'{args.data}_{args.loss}.txt', 'a') as f:
    #     f.write(f'Training Accuracy: {accuracy} \t Testing Accuracy: {test_accuracy} \t Training Loss: {loss} \t Testing Loss: {test_loss} \n')

# test_accuracy, test_loss = test(model, device, test_data, test_label, test_original, credit, args.batch_size, args.loss)
# writer.add_scalar('Accuarcy: ', test_accuracy, number)
# writer.add_scalar('Loss: ', test_loss, number)
# print(test_accuracy, test_loss, number)
    

