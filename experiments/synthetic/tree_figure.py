import argparse
parser = argparse.ArgumentParser(description='create tree figures')
parser.add_argument('--experiment', choices=['loss_vs_structure', 'base_experiment', 'para_norm'], required=True)
# parser.add_argument('--loss', choices=['xentropy', 'tree', 'simloss', 'HSM'], required=True)
args = parser.parse_args()

import matplotlib.pyplot as plt
import numpy as np

f_tree = open(f'{args.experiment}_tree_loss_1.txt', 'r')
f_xentropy = open(f'{args.experiment}_xentropy_loss_1.txt', 'r')
# f_simloss = open(f'{args.experiment}_simloss.txt', 'r')
# f_HSM = open(f'{args.experiment}_HSM.txt', 'r')

if args.experiment == 'loss_vs_structure':
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
    for num in range(11):
        loss_mid = []
        W_err_mid = []
        accuracy_mid = []
        for i in range(50):
            loss_mid.append(loss_tree_all[num+i*11])
            W_err_mid.append(W_err_tree_all[num+i*11])
            accuracy_mid.append(accuracy_tree_all[num+i*11])
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

            loss = float(line[loss_end:W_err_start].strip())
            W_err = float(line[W_err_end:accuracy_start].strip())
            accuracy = float(float(line[accuracy_end:].strip()))

            loss_xentropy_all.append(loss)
            W_err_xentropy_all.append(W_err)
            accuracy_xentropy_all.append(accuracy)
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

            loss = float(line[loss_end:W_err_start].strip())
            W_err = float(line[W_err_end:accuracy_start].strip())
            accuracy = float(float(line[accuracy_end:].strip()))

            loss_simloss_all.append(loss)
            W_err_simloss_all.append(W_err)
            accuracy_simloss_all.append(accuracy)
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

            loss = float(line[loss_end:W_err_start].strip())
            W_err = float(line[W_err_end:accuracy_start].strip())
            accuracy = float(float(line[accuracy_end:].strip()))

            loss_HSM_all.append(loss)
            W_err_HSM_all.append(W_err)
            accuracy_HSM_all.append(accuracy)
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

    accuracy_ = accuracy_tree[:10]
    accuracy_.append(0.098)
    accuracy_xentropy.append(accuracy_xentropy[0])
    x = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    plt.figure(0)
    l1, = plt.plot(x, accuracy_)
    l2, = plt.plot(x, accuracy_xentropy, linestyle='-.')
    # l3, = plt.plot(x, accuracy_simloss, linestyle='--')
    # l4, = plt.plot(x, accuracy_HSM, linestyle='dotted')
    plt.legend(handles=[l1,l2],labels=['Cover Tree Loss', 'Cross Entropy Loss'])
    plt.xlabel('Randomness of Cover Tree Structure')
    plt.ylabel('Accuracy')
    plt.savefig('loss_vs_structure.png', dpi=300)


if args.experiment == 'base_experiment':
    train_loss_all = []
    train_accuracy_all = []
    height_all = []
    test_loss_all = []
    test_accuracy_all = []
    for line in f_tree:
        line = line.strip()
        if len(line)>1:

            loss_start = line.find('Loss:')
            loss_end  = loss_start + len('Loss:')

            accuracy_start = line.find('Accuracy:')
            accuracy_end = accuracy_start + len('Accuracy:')

            # height_start = line.find('Height:')
            # height_end = height_start + len('Height:')

            test_loss_start = line.find('Generlization Loss:')
            test_loss_end = test_loss_start + len('Generlization Loss:')

            test_accuracy_start = line.find('Test Accuracy:')
            test_accuracy_end = test_accuracy_start + len('Test Accuracy:')

            loss = float(line[loss_end:accuracy_start].strip())
            accuracy = float(line[accuracy_end:test_loss_start].strip())
            # height = int(line[height_end:test_loss_start].strip())
            test_loss = float(line[test_loss_end:test_accuracy_start].strip())
            test_accuracy = float(line[test_accuracy_end:].strip())

            train_loss_all.append(loss)
            train_accuracy_all.append(accuracy)
            # height_all.append(height)
            test_loss_all.append(test_loss)
            test_accuracy_all.append(test_accuracy)
    train_loss_tree = []
    train_accuracy_tree = []
    height_tree = []
    test_loss_tree = []
    test_accuracy_tree = []
    for num in range(24):
        loss_mid = []
        accuracy_mid = []
        height_mid = []
        test_loss_mid = []
        test_accuracy_mid = []
        for i in range(50):
            loss_mid.append(train_loss_all[num+i*24])
            accuracy_mid.append(train_accuracy_all[num+i*24])
            # height_mid.append(height_all[num+i*24])
            test_loss_mid.append(test_loss_all[num+i*24])
            test_accuracy_mid.append(test_accuracy_all[num+i*24])

        train_loss_tree.append(np.mean(loss_mid))
        train_accuracy_tree.append(np.mean(accuracy_mid))
        # height_tree.append(int(np.mean(height_mid)))
        test_loss_tree.append(np.mean(test_loss_mid))
        test_accuracy_tree.append(np.mean(test_accuracy_mid))

    train_loss_all = []
    train_accuracy_all = []
    test_loss_all = []
    test_accuracy_all = []
    for line in f_xentropy:
        line = line.strip()
        if len(line)>1:

            loss_start = line.find('Loss:')
            loss_end  = loss_start + len('Loss:')

            accuracy_start = line.find('Accuracy:')
            accuracy_end = accuracy_start + len('Accuracy:')

            test_loss_start = line.find('Generlization Loss:')
            test_loss_end = test_loss_start + len('Generlization Loss:')

            test_accuracy_start = line.find('Test Accuracy:')
            test_accuracy_end = test_accuracy_start + len('Test Accuracy:')

            loss = float(line[loss_end:accuracy_start].strip())
            accuracy = float(line[accuracy_end:test_loss_start].strip())
            test_loss = float(line[test_loss_end:test_accuracy_start].strip())
            test_accuracy = float(line[test_accuracy_end:].strip())

            train_loss_all.append(loss)
            train_accuracy_all.append(accuracy)
            test_loss_all.append(test_loss)
            test_accuracy_all.append(test_accuracy)
    train_loss_xentropy = []
    train_accuracy_xentropy = []
    test_loss_xentropy = []
    test_accuracy_xentropy = []
    for num in range(24):
        loss_mid = []
        accuracy_mid = []
        test_loss_mid = []
        test_accuracy_mid = []
        for i in range(50):
            loss_mid.append(train_loss_all[num+i*24])
            accuracy_mid.append(train_accuracy_all[num+i*24])
            test_loss_mid.append(test_loss_all[num+i*24])
            test_accuracy_mid.append(test_accuracy_all[num+i*24])

        train_loss_xentropy.append(np.mean(loss_mid))
        train_accuracy_xentropy.append(np.mean(accuracy_mid))
        test_loss_xentropy.append(np.mean(test_loss_mid))
        test_accuracy_xentropy.append(np.mean(test_accuracy_mid))

    # train_accuracy_xentropy = []
    # for i in range(24):
    #     train_accuracy_xentropy.append(0.027)
    #     i += 1
    # train_loss_tree = train_loss_tree[3:]
    # test_loss_tree = test_loss_tree[3:]
    # train_accuracy_tree = train_accuracy_tree[3:]
    # height_tree = height_tree[3:]


    plt.figure(0)
    # base = np.arange(1.1,3.1,0.1)
    base = [1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.0,4.0,5.0,6.0,7.0]
    l1, = plt.plot(base, train_accuracy_tree)
    # l2, = plt.plot(base, test_accuracy_tree, linestyle="-.")
    l2, = plt.plot(base, train_accuracy_xentropy, linestyle='--')
    # l4, = plt.plot(base, test_accuracy_xentropy, linestyle='dotted')
    plt.legend(handles=[l1,l2],labels=['Cover Tree Loss', 'Cross Entropy Loss'])
    plt.xlabel('Base of Cover Tree')
    plt.ylabel('Accuracy')
    plt.savefig('accuracy_vs_base.png', dpi=300)

    # plt.figure(1)
    # # base = np.arange(1.1,3.1,0.1)
    # base = [1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.0,4.0,5.0,6.0,7.0]
    # l1, = plt.plot(base, height_tree)
    # # plt.legend(handles=[l1],labels=['Train Accuracy', 'Test Accuracy'])
    # plt.xlabel('Base of Cover Tree')
    # plt.ylabel('Height')
    # plt.savefig('height_vs_base.png', dpi=300)

    plt.figure(2)
    # base = np.arange(1.1,3.1,0.1)
    base = [1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.0,4.0,5.0,6.0,7.0]
    l1, = plt.plot(base, train_loss_tree)
    l2, = plt.plot(base, test_loss_tree)
    plt.legend(handles=[l1,l2],labels=['Train Loss', 'Test Loss'])
    plt.xlabel('Base of Cover Tree')
    plt.ylabel('Loss')
    plt.savefig('loss_vs_base.png', dpi=300)


    
if args.experiment == 'para_norm':
    level = np.arange(0,9,1)
    W_norm = [0.5535,0.3836,0.1933,0.0927,0.0441,0.0262,0.0237,0.0356,0.1033]
    # W_norm = [0.0175, 0.0175, 0.0187, 0.0185, 0.0183, 0.0185, 0.0211, 0.0345, 0.1033]
    l1, = plt.plot(level, W_norm)
    plt.xlabel('Level of Cover Tree')
    plt.ylabel('|W|')
    plt.savefig('W_vs_level.png', dpi=300)
    