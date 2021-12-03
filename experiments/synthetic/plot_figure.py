import argparse
parser = argparse.ArgumentParser(description='create figures')
parser.add_argument('--experiment', choices=['loss_vs_n','loss_vs_d', 'loss_vs_sigma', 'loss_vs_c'], required=True)
args = parser.parse_args()

import matplotlib.pyplot as plt
import numpy as np
import math

# f_tree = open(f'{args.experiment}_tree_original.txt', 'r')
# f_xentropy = open(f'{args.experiment}_xentropy_original.txt', 'r')
# f_simloss_1 = open(f'{args.experiment}_simloss_original.txt', 'r')
# f_simloss_2 = open(f'{args.experiment}_simloss_original_0.6.txt', 'r')
# f_simloss_3 = open(f'{args.experiment}_simloss_original_0.7.txt', 'r')
# f_simloss_4 = open(f'{args.experiment}_simloss_original_0.8.txt', 'r')
# f_simloss_5 = open(f'{args.experiment}_simloss_original_0.9.txt', 'r')
# f_simloss = [f_simloss_1, f_simloss_2, f_simloss_3, f_simloss_4, f_simloss_5]
# f_HSM = open(f'{args.experiment}_HSM_original.txt', 'r')

f_tree = open(f'{args.experiment}_tree_0.0001.txt','r')
f_xentropy = open(f'{args.experiment}_xentropy_0.0001.txt', 'r')
f_simloss = open(f'{args.experiment}_simloss_0.0001.txt', 'r')
f_HSM = open(f'{args.experiment}_HSM_0.0001.txt', 'r')

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

        loss_ = float(line[loss_end:W_err_start].strip())
        W_err_ = float(line[W_err_end:accuracy_start].strip())
        accuracy_ = float(float(line[accuracy_end:].strip()))

        loss_simloss_all.append(loss_)
        W_err_simloss_all.append(W_err_)
        accuracy_simloss_all.append(accuracy_)
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

# loss_simloss = []
# W_err_simloss = []
# accuracy_simloss = []
# for i in f_simloss:
#     j = 1
#     loss_simloss_all = []
#     W_err_simloss_all = []
#     accuracy_simloss_all = []
#     for line in i:
#         line = line.strip()
#         if len(line)>1:

#             loss_start = line.find('Loss:')
#             loss_end  = loss_start + len('Loss:')

#             W_err_start = line.find('W_err:')
#             W_err_end = W_err_start + len('W_err:')

#             accuracy_start = line.find('Accuracy:')
#             accuracy_end = accuracy_start + len('Accuracy:')

#             loss__ = float(line[loss_end:W_err_start].strip())
#             W_err__ = float(line[W_err_end:accuracy_start].strip())
#             accuracy__ = float(float(line[accuracy_end:].strip()))

#             loss_simloss_all.append(loss__)
#             W_err_simloss_all.append(W_err__)
#             accuracy_simloss_all.append(accuracy__)
#     loss_simloss_ = []
#     W_err_simloss_ = []
#     accuracy_simloss_ = []
#     for num in range(10):
#         loss_mid = []
#         W_err_mid = []
#         accuracy_mid = []
#         for i in range(50):
#             loss_mid.append(loss_simloss_all[num+i*10])
#             W_err_mid.append(W_err_simloss_all[num+i*10])
#             accuracy_mid.append(accuracy_simloss_all[num+i*10])
#         loss_simloss_.append(np.mean(loss_mid))
#         W_err_simloss_.append(np.mean(W_err_mid))
#         accuracy_simloss_.append(np.mean(accuracy_mid))
#     loss_simloss.append(loss_simloss_)
#     W_err_simloss.append(W_err_simloss_)
#     accuracy_simloss.append(accuracy_simloss_)

loss_HSM_all = []
W_err_HSM_all = []
accuracy_HSM_all = []
for line in f_HSM:
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

        loss_HSM_all.append(loss_)
        W_err_HSM_all.append(W_err_)
        accuracy_HSM_all.append(accuracy_)
loss_HSM = []
W_err_HSM = []
accuracy_HSM = []
for num in range(10):
    loss_mid = []
    W_err_mid = []
    accuracy_mid = []
    for i in range(50):
        loss_mid.append(loss_HSM_all[num+i*10])
        W_err_mid.append(W_err_HSM_all[num+i*10])
        accuracy_mid.append(accuracy_HSM_all[num+i*10])
    loss_HSM.append(np.mean(loss_mid))
    W_err_HSM.append(np.mean(W_err_mid))
    accuracy_HSM.append(np.mean(accuracy_mid))

# W_diff_xentropy = np.array(W_err_xentropy) - np.array(W_err_tree)
# W_diff_simloss_0 = np.array(W_err_simloss[0]) - np.array(W_err_tree) 
# W_diff_simloss_1 = np.array(W_err_simloss[1]) - np.array(W_err_tree) 
# W_diff_simloss_2 = np.array(W_err_simloss[2]) - np.array(W_err_tree) 
# W_diff_simloss_3 = np.array(W_err_simloss[3]) - np.array(W_err_tree) 
# W_diff_simloss_4 = np.array(W_err_simloss[4]) - np.array(W_err_tree) 
# W_diff_tree = np.array(W_err_tree) - np.array(W_err_tree)

# W_diff_xentropy = [math.log(y+1e-5) for y in W_diff_xentropy]
# W_diff_simloss = [math.log(y+1e-5) for y in W_diff_simloss]
# W_diff_tree = [math.log(y+1e-5) for y in W_diff_tree]

# loss_tree = [math.log(y) for y in loss_tree]
# W_err_tree = [math.log(y) for y in W_err_tree]
# accuracy_tree = [math.log(y) for y in accuracy_tree]

# loss_xentropy = [math.log(y) for y in loss_xentropy]
# W_err_xentropy = [math.log(y) for y in W_err_xentropy]
# accuracy_xentropy = [math.log(y) for y in accuracy_xentropy]

# loss_simloss = [math.log(y) for y in loss_simloss]
# W_err_simloss = [math.log(y) for y in W_err_simloss]
# accuracy_simloss = [math.log(y) for y in accuracy_simloss]

if args.experiment == 'loss_vs_n' :
    x = [16,32,64,128,256,512,1024,2048,4096,8192]
    plt.figure(0)
    l1, = plt.loglog(x, loss_tree)
    l2, = plt.loglog(x, loss_xentropy, linestyle="-.")
    # l3, = plt.loglog(x, loss_simloss[0], 'r', linestyle="--")
    # l4, = plt.loglog(x, loss_simloss[1], 'b', linestyle="--")
    # l5, = plt.loglog(x, loss_simloss[2], 'g', linestyle="--")
    # l6, = plt.loglog(x, loss_simloss[3], 'orange', linestyle="--")
    l3, = plt.loglog(x, loss_simloss, linestyle="--")
    l4, = plt.loglog(x, loss_HSM, linestyle="dotted")
    plt.legend(handles=[l1,l2,l3,l4],labels=['Tree Loss','Cross Entropy Loss','SimLoss', 'HSM'])
    plt.xlabel('Number of Data Points($n$)', fontsize=15)
    plt.ylabel('Loss', fontsize=15)
    plt.savefig('loss_vs_n.png', dpi=300)

    plt.figure(1)
    l1, = plt.loglog(x, W_err_tree,)
    l2, = plt.loglog(x, W_err_xentropy, linestyle="-.")
    # l3, = plt.loglog(x, W_err_simloss[0], 'r', linestyle="--")
    # l4, = plt.loglog(x, W_err_simloss[1], 'b', linestyle="--")
    # l5, = plt.loglog(x, W_err_simloss[2], 'g', linestyle="--")
    # l6, = plt.loglog(x, W_err_simloss[3], 'orange', linestyle="--")
    l3, = plt.loglog(x, W_err_simloss, linestyle="--")
    l4, = plt.loglog(x, W_err_HSM, linestyle="dotted")
    plt.legend(handles=[l1,l2,l3,l4],labels=['Tree Loss','Cross Entropy Loss','SimLoss', 'HSM'])
    plt.xlabel('Number of Data Points($n$)')
    plt.ylabel('|W_error|')
    plt.savefig('error_vs_n.png', dpi=300)

    plt.figure(2, figsize=(10,5))
    l1, = plt.loglog(x, accuracy_tree)
    l2, = plt.loglog(x, accuracy_xentropy, linestyle="-.")
    # l3, = plt.loglog(x, accuracy_simloss[0],'r', linestyle="--")
    # l4, = plt.loglog(x, accuracy_simloss[1], 'b', linestyle="--")
    # l5, = plt.loglog(x, accuracy_simloss[2], 'g', linestyle="--")
    # l6, = plt.loglog(x, accuracy_simloss[3],'orange', linestyle="--")
    l3, = plt.loglog(x, accuracy_simloss, linestyle="--")
    l4, = plt.loglog(x, accuracy_HSM, linestyle="dotted")
    plt.legend(handles=[l1,l2,l3,l4],labels=['Tree Loss','Cross Entropy Loss','SimLoss', 'HSM'],fontsize=15)
    plt.xlabel('Number of Data Points($n$)', fontsize=15)
    plt.ylabel('Accuracy',fontsize=15)
    plt.savefig('accuracy_vs_n.png', dpi=300)


if args.experiment == 'loss_vs_d':
    x = [2,4,8,16,32,64,128,256,512,1024]
    plt.figure(0)
    l1, = plt.loglog(x, loss_tree)
    l2,= plt.loglog(x, loss_xentropy, linestyle="-.")
    # l3, = plt.loglog(x, loss_simloss[0], 'r', linestyle="--")
    # l4, = plt.loglog(x, loss_simloss[1], 'b', linestyle="--")
    # l5, = plt.loglog(x, loss_simloss[2], 'g', linestyle="--")
    # l6, = plt.loglog(x, loss_simloss[3], 'orange', linestyle="--")
    l3, = plt.loglog(x, loss_simloss, linestyle="--")
    l4, = plt.loglog(x, loss_HSM, linestyle="dotted")
    plt.legend(handles=[l1,l2,l3,l4],labels=['Tree Loss','Cross Entropy Loss', 'SimLoss','HSM'])
    plt.xlabel('Dimension($d$)')
    plt.ylabel('Loss')
    plt.savefig('loss_vs_d.png', dpi=300)

    plt.figure(1)
    l1, = plt.loglog(x, W_err_tree)
    l2, = plt.loglog(x, W_err_xentropy, linestyle="-.")
    # l3, = plt.loglog(x, W_diff_simloss_0, 'r', linestyle="--")
    # l4, = plt.loglog(x, W_diff_simloss_1, 'b', linestyle="--")
    # l5, = plt.loglog(x, W_diff_simloss_2, 'g', linestyle="--")
    # l6, = plt.loglog(x, W_diff_simloss_3, 'orange', linestyle="--")
    l3, = plt.loglog(x, W_err_simloss, linestyle="--")
    l4, = plt.loglog(x, W_err_HSM, linestyle="dotted")
    plt.legend(handles=[l1,l2,l3,l4],labels=['Tree Loss','Cross Entropy Loss', 'SimLoss','HSM'])
    plt.xlabel('Dimension($d$)')
    plt.ylabel('|W_error|')
    plt.savefig('error_vs_d.png', dpi=300)

    plt.figure(2, figsize=(10,5))
    l1, = plt.loglog(x, accuracy_tree)
    l2, = plt.loglog(x, accuracy_xentropy, linestyle="-.")
    # l3, = plt.loglog(x, accuracy_simloss[0], 'r', linestyle="--")
    # l4, = plt.loglog(x, accuracy_simloss[1], 'b', linestyle="--")
    # l5, = plt.loglog(x, accuracy_simloss[2], 'g', linestyle="--")
    # l6, = plt.loglog(x, accuracy_simloss[3], 'orange', linestyle="--")
    l3, = plt.loglog(x, accuracy_simloss, linestyle="--")
    l4, = plt.loglog(x, accuracy_HSM, linestyle="dotted")
    plt.legend(handles=[l1,l2,l3,l4],labels=['Tree Loss','Cross Entropy Loss', 'SimLoss','HSM'], fontsize=15)
    plt.xlabel('Dimension($d$)', fontsize=15)
    plt.ylabel('Accuracy', fontsize=15)
    plt.savefig('accuracy_vs_d.png', dpi=300)

if args.experiment == 'loss_vs_sigma':
    x = [1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3.0,3.25]
    plt.figure(0)
    l1, = plt.loglog(x, loss_tree)
    l2, = plt.loglog(x, loss_xentropy, linestyle="-.")
    # l3, = plt.loglog(x, loss_simloss[0], 'r', linestyle="--")
    # l4, = plt.loglog(x, loss_simloss[1], 'b', linestyle="--")
    # l5, = plt.loglog(x, loss_simloss[2], 'g', linestyle="--")
    # l6, = plt.loglog(x, loss_simloss[3], 'orange', linestyle="--")
    l3, = plt.loglog(x, loss_simloss, 'gray', linestyle="--")
    l4, = plt.loglog(x, loss_HSM, linestyle="dotted")
    plt.legend(handles=[l1,l2,l3,l4],labels=['Tree Loss','Cross Entropy Loss', 'SimLoss','HSM'])
    plt.xlabel('Randomness($\sigma$)')
    plt.ylabel('Loss')
    plt.savefig('loss_vs_sigma.png', dpi=300)

    plt.figure(1)
    l1, = plt.loglog(x, W_err_tree)
    l2, = plt.loglog(x, W_err_xentropy, linestyle="-.")
    # l3, = plt.loglog(x, W_err_simloss[0], 'r', linestyle="--")
    # l4, = plt.loglog(x, W_err_simloss[1], 'b', linestyle="--")
    # l5, = plt.loglog(x, W_err_simloss[2], 'g', linestyle="--")
    # l6, = plt.loglog(x, W_err_simloss[3], 'orange', linestyle="--")
    l3, = plt.loglog(x, W_err_simloss, linestyle="--")
    l4, = plt.loglog(x, W_err_HSM, linestyle="dotted")
    plt.legend(handles=[l1,l2,l3,l4],labels=['Tree Loss','Cross Entropy Loss', 'SimLoss','HSM'])
    plt.xlabel('Randomness($\sigma$)')
    plt.ylabel('|W_error|')
    plt.savefig('error_vs_sigma.png', dpi=300)

    plt.figure(2, figsize=(10,5))
    l1, = plt.loglog(x, accuracy_tree)
    l2, = plt.loglog(x, accuracy_xentropy, linestyle="-.")
    # l3, = plt.loglog(x, accuracy_simloss[0], 'r', linestyle="--")
    # l4, = plt.loglog(x, accuracy_simloss[1], 'b', linestyle="--")
    # l5, = plt.loglog(x, accuracy_simloss[2], 'g', linestyle="--")
    # l6, = plt.loglog(x, accuracy_simloss[3], 'orange', linestyle="--")
    l3, = plt.loglog(x, accuracy_simloss, linestyle="--")
    l4, = plt.loglog(x, accuracy_HSM, linestyle="dotted")
    plt.legend(handles=[l1,l2,l3,l4],labels=['Tree Loss','Cross Entropy Loss', 'SimLoss','HSM'], fontsize=15)
    plt.xlabel('Randomness($\sigma$)', fontsize=15)
    plt.ylabel('Accuracy', fontsize=15)
    plt.savefig('accuracy_vs_sigma.png', dpi=300)

if args.experiment == 'loss_vs_c':
    x = [10,20,30,40,50,60,70,80,90,100]
    plt.figure(0)
    l1, = plt.loglog(x, loss_tree)
    l2, = plt.loglog(x, loss_xentropy, linestyle="-.")
    # l3, = plt.loglog(x, loss_simloss[0], 'gray', linestyle="--")
    l3, = plt.loglog(x, loss_simloss, linestyle="--")
    # l5, = plt.loglog(x, loss_simloss[2], 'gray', linestyle="--")
    # l6, = plt.loglog(x, loss_simloss[3], 'gray', linestyle="--")
    # l7, = plt.loglog(x, loss_simloss[4], 'gray', linestyle="--")
    l4, = plt.loglog(x, loss_HSM, linestyle="dotted")
    plt.legend(handles=[l1,l2,l3,l4],labels=['Tree Loss','Cross Entropy Loss', 'SimLoss','HSM'])
    plt.xlabel('Number of Classes($k$)')
    plt.ylabel('Loss')
    plt.savefig('loss_vs_class.png', dpi=300)

    plt.figure(1)
    l1, = plt.loglog(x, W_err_tree)
    l2, = plt.loglog(x, W_err_xentropy, linestyle="-.")
    # l3, = plt.loglog(x, W_diff_simloss_0, 'gray', linestyle="--")
    l3, = plt.loglog(x, W_err_simloss, linestyle="--")
    # l5, = plt.loglog(x, W_diff_simloss_2, 'gray', linestyle="--")
    # l6, = plt.loglog(x, W_diff_simloss_3, 'gray', linestyle="--")
    # l7, = plt.loglog(x, W_diff_simloss_4, 'gray', linestyle="--")
    l4, = plt.loglog(x, W_err_HSM, linestyle="dotted")
    plt.legend(handles=[l1,l2,l3,l4],labels=['Tree Loss','Cross Entropy Loss', 'SimLoss','HSM'])
    plt.xlabel('Number of Classes($k$)')
    plt.ylabel('|W_error|')
    plt.savefig('error_vs_class.png', dpi=300)

    plt.figure(2, figsize=(10,5))
    l1, = plt.loglog(x, accuracy_tree)
    l2, = plt.loglog(x, accuracy_xentropy, linestyle="-.")
    # l3, = plt.loglog(x, accuracy_simloss[0], 'r', linestyle="--")
    l3, = plt.loglog(x, accuracy_simloss, linestyle="--")
    # l5, = plt.loglog(x, accuracy_simloss[2], 'g', linestyle="--")
    # l6, = plt.loglog(x, accuracy_simloss[3], 'orange', linestyle="--")
    # l7, = plt.loglog(x, accuracy_simloss[4], 'gray', linestyle="--")
    l4, = plt.loglog(x, accuracy_HSM, linestyle="dotted")
    plt.legend(handles=[l1,l2,l3,l4],labels=['Tree Loss','Cross Entropy Loss', 'SimLoss','HSM'], fontsize=15)
    plt.xlabel('Number of Classes ($k$)', fontsize=15)
    plt.ylabel('Accuracy', fontsize=15)
    plt.savefig('accuracy_vs_class.png', dpi=300)