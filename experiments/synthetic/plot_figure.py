import argparse
parser = argparse.ArgumentParser(description='create figures')
parser.add_argument('--experiment', choices=['loss_vs_n','loss_vs_d', 'loss_vs_sigma', 'loss_vs_c'], required=True)
args = parser.parse_args()

import matplotlib.pyplot as plt
import numpy as np

f_tree = open(f'{args.experiment}_tree.txt', 'r')
f_xentropy = open(f'{args.experiment}_xentropy.txt', 'r')
f_simloss = open(f'{args.experiment}_simloss.txt', 'r')

loss_tree_all = []
W_err_tree_all = []
accuracy_tree_all = []
for line in f_tree:
    line = line.strip()
    if len(line)>1:

        loss_start = line.find('Loss:')
        loss_end  = loss_start + len('Loss:')

        W_err_start = line.find('W_err:')
        W_err_end = W_err_start + len('W_err:')

        accuracy_start = line.find('Accuracy:')
        accuracy_end = accuracy_start + len('Accuracy:')

        loss = float(line[loss_end:W_err_start].strip())
        W_err = float(line[W_err_end:accuracy_start].strip())
        accuracy = float(float(line[accuracy_end:].strip()))

        loss_tree_all.append(loss)
        W_err_tree_all.append(W_err)
        accuracy_tree_all.append(accuracy)
loss_tree = []
W_err_tree = []
accuracy_tree = []
for num in range(10):
    loss_mid = []
    W_err_mid = []
    accuracy_mid = []
    for i in range(50):
        loss_mid.append(loss_tree_all[num+i*10])
        W_err_mid.append(W_err_tree_all[num+i*10])
        accuracy_mid.append(accuracy_tree_all[num+i*10])
    loss_tree.append(np.mean(loss_mid))
    W_err_tree.append(np.mean(W_err_mid))
    accuracy_tree.append(np.mean(accuracy_mid))

loss_xentropy_all = []
W_err_xentropy_all = []
accuracy_xentropy_all = []
for line in f_xentropy:
    line = line.strip()
    if len(line)>1:

        loss_start = line.find('Loss:')
        loss_end  = loss_start + len('Loss:')

        W_err_start = line.find('W_err:')
        W_err_end = W_err_start + len('W_err:')

        accuracy_start = line.find('Accuracy:')
        accuracy_end = accuracy_start + len('Accuracy:')

        loss_ = float(line[loss_end:W_err_start].strip())
        W_err_ = float(line[W_err_end:accuracy_start].strip())
        accuracy_ = float(float(line[accuracy_end:].strip()))

        loss_xentropy_all.append(loss_)
        W_err_xentropy_all.append(W_err_)
        accuracy_xentropy_all.append(accuracy_)
loss_xentropy = []
W_err_xentropy = []
accuracy_xentropy = []
for num in range(10):
    loss_mid = []
    W_err_mid = []
    accuracy_mid = []
    for i in range(50):
        loss_mid.append(loss_xentropy_all[num+i*10])
        W_err_mid.append(W_err_xentropy_all[num+i*10])
        accuracy_mid.append(accuracy_xentropy_all[num+i*10])
    loss_xentropy.append(np.mean(loss_mid))
    W_err_xentropy.append(np.mean(W_err_mid))
    accuracy_xentropy.append(np.mean(accuracy_mid))

loss_simloss_all = []
W_err_simloss_all = []
accuracy_simloss_all = []
for line in f_simloss:
    line = line.strip()
    if len(line)>1:

        loss_start = line.find('Loss:')
        loss_end  = loss_start + len('Loss:')

        W_err_start = line.find('W_err:')
        W_err_end = W_err_start + len('W_err:')

        accuracy_start = line.find('Accuracy:')
        accuracy_end = accuracy_start + len('Accuracy:')

        loss__ = float(line[loss_end:W_err_start].strip())
        W_err__ = float(line[W_err_end:accuracy_start].strip())
        accuracy__ = float(float(line[accuracy_end:].strip()))

        loss_simloss_all.append(loss__)
        W_err_simloss_all.append(W_err__)
        accuracy_simloss_all.append(accuracy__)
loss_simloss = []
W_err_simloss = []
accuracy_simloss = []
for num in range(10):
    loss_mid = []
    W_err_mid = []
    accuracy_mid = []
    for i in range(50):
        loss_mid.append(loss_simloss_all[num+i*10])
        W_err_mid.append(W_err_simloss_all[num+i*10])
        accuracy_mid.append(accuracy_simloss_all[num+i*10])
    loss_simloss.append(np.mean(loss_mid))
    W_err_simloss.append(np.mean(W_err_mid))
    accuracy_simloss.append(np.mean(accuracy_mid))

if args.experiment == 'loss_vs_n' :
    x = [16,32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    plt.figure(0)
    l1, = plt.plot(x, loss_tree)
    l2, = plt.plot(x, loss_xentropy, linestyle="-.")
    l3, = plt.plot(x, loss_simloss, linestyle="--")
    plt.legend(handles=[l1,l2,l3],labels=['Cover Tree Loss','Cross Entropy Loss', 'SimLoss'])
    plt.xlabel('Number of Data Points')
    plt.ylabel('Loss')
    plt.savefig('loss_vs_n.png', dpi=300)

    plt.figure(1)
    l1, = plt.plot(x, W_err_tree)
    l2, = plt.plot(x, W_err_xentropy, linestyle="-.")
    l3, = plt.plot(x, W_err_simloss, linestyle="--")
    plt.legend(handles=[l1,l2,l3],labels=['Cover Tree Loss','Cross Entropy Loss', 'SimLoss'])
    plt.xlabel('Number of Data Points')
    plt.ylabel('|W_error|')
    plt.savefig('error_vs_n.png', dpi=300)

    plt.figure(2)
    l1, = plt.plot(x, accuracy_tree)
    l2, = plt.plot(x, accuracy_xentropy, linestyle="-.")
    l3, = plt.plot(x, accuracy_simloss, linestyle="--")
    plt.legend(handles=[l1,l2,l3],labels=['Cover Tree Loss','Cross Entropy Loss', 'SimLoss'])
    plt.xlabel('Number of Data Points')
    plt.ylabel('Accuracy')
    plt.savefig('accuracy_vs_n.png', dpi=300)


if args.experiment == 'loss_vs_d':
    x = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    plt.figure(0)
    l1, = plt.plot(x, loss_tree)
    l2,= plt.plot(x, loss_xentropy, linestyle="-.")
    l3, = plt.plot(x, loss_simloss, linestyle="--")
    plt.legend(handles=[l1,l2,l3],labels=['Cover Tree Loss','Cross Entropy Loss', 'SimLoss'])
    plt.xlabel('Dimension')
    plt.ylabel('Loss')
    plt.savefig('loss_vs_d.png', dpi=300)

    diff_xentropy = np.array(W_err_tree) - np.array(W_err_xentropy)
    diff_simloss = np.array(W_err_tree) - np.array(W_err_simloss)
    print(diff_xentropy)
    print(diff_simloss)
    plt.figure(1)
    l1, = plt.plot(x, W_err_tree)
    l2, = plt.plot(x, diff_xentropy, linestyle="-.")
    l3, = plt.plot(x, diff_simloss, linestyle="--")
    plt.legend(handles=[l1,l2,l3],labels=['Cover Tree Loss','Cross Entropy Loss', 'SimLoss'])
    plt.xlabel('Dimension')
    plt.ylabel('|W_error|')
    plt.savefig('error_vs_d.png', dpi=300)

    plt.figure(2)
    l1, = plt.plot(x, accuracy_tree)
    l2, = plt.plot(x, accuracy_xentropy, linestyle="-.")
    l3, = plt.plot(x, accuracy_simloss, linestyle="--")
    plt.legend(handles=[l1,l2,l3],labels=['Cover Tree Loss','Cross Entropy Loss', 'SimLoss'])
    plt.xlabel('Number of Data Points')
    plt.ylabel('Accuracy')
    plt.savefig('accuracy_vs_d.png', dpi=300)

if args.experiment == 'loss_vs_sigma':
    x = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25]
    plt.figure(0)
    l1, = plt.plot(x, loss_tree)
    l2, = plt.plot(x, loss_xentropy, linestyle="-.")
    l3, = plt.plot(x, loss_simloss, linestyle="--")
    plt.legend(handles=[l1,l2,l3],labels=['Cover Tree Loss','Cross Entropy Loss', 'SimLoss'])
    plt.xlabel('Randomness')
    plt.ylabel('Loss')
    plt.savefig('loss_vs_sigma.png', dpi=300)

    plt.figure(1)
    l1, = plt.plot(x, W_err_tree)
    l2, = plt.plot(x, W_err_xentropy, linestyle="-.")
    l3, = plt.plot(x, W_err_simloss, linestyle="--")
    plt.legend(handles=[l1,l2,l3],labels=['Cover Tree Loss','Cross Entropy Loss', 'SimLoss'])
    plt.xlabel('Randomness')
    plt.ylabel('|W_error|')
    plt.savefig('error_vs_sigma.png', dpi=300)

    plt.figure(2)
    l1, = plt.plot(x, accuracy_tree)
    l2, = plt.plot(x, accuracy_xentropy, linestyle="-.")
    l3, = plt.plot(x, accuracy_simloss, linestyle="--")
    plt.legend(handles=[l1,l2,l3],labels=['Cover Tree Loss','Cross Entropy Loss', 'SimLoss'])
    plt.xlabel('Randomness')
    plt.ylabel('Accuracy')
    plt.savefig('accuracy_vs_sigma.png', dpi=300)

if args.experiment == 'loss_vs_c':
    x = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    plt.figure(0)
    l1, = plt.plot(x, loss_tree)
    l2, = plt.plot(x, loss_xentropy, linestyle="-.")
    l3, = plt.plot(x, loss_simloss, linestyle="--")
    plt.legend(handles=[l1,l2,l3],labels=['Cover Tree Loss','Cross Entropy Loss', 'SimLoss'])
    plt.xlabel('Number of Classes')
    plt.ylabel('Loss')
    plt.savefig('loss_vs_class.png', dpi=300)

    diff_xentropy = np.array(W_err_tree) - np.array(W_err_xentropy)
    diff_simloss = np.array(W_err_tree) - np.array(W_err_simloss)
    print(diff_xentropy)
    print(diff_simloss)
    plt.figure(1)
    l1, = plt.plot(x, W_err_tree)
    l2, = plt.plot(x, diff_xentropy, linestyle="-.")
    l3, = plt.plot(x, diff_simloss, linestyle="--")
    plt.legend(handles=[l1,l2,l3],labels=['Cover Tree Loss','Cross Entropy Loss', 'SimLoss'])
    plt.xlabel('Number of Classes')
    plt.ylabel('|W_error|')
    plt.savefig('error_vs_class.png', dpi=300)

    plt.figure(2)
    l1, = plt.plot(x, accuracy_tree)
    l2, = plt.plot(x, accuracy_xentropy, linestyle="-.")
    l3, = plt.plot(x, accuracy_simloss, linestyle="--")
    plt.legend(handles=[l1,l2,l3],labels=['Cover Tree Loss','Cross Entropy Loss', 'SimLoss'])
    plt.xlabel('Number of Classes')
    plt.ylabel('Accuracy')
    plt.savefig('accuracy_vs_class.png', dpi=300)