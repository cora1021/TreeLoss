import matplotlib.pyplot as plt
import numpy as np

f_tree1 = open(f'1_tree_100_1000.txt', 'r')
f_tree2 = open(f'2_tree_100_1000.txt', 'r')
f_tree3 = open(f'3_tree_100_1000.txt', 'r')
f_tree4 = open(f'4_tree_100_1000.txt', 'r')
f_tree5 = open(f'5_tree_100_1000.txt', 'r')
f_xentropy = open('1_xentropy_100_1000.txt','r')

accuracy_tree = []
accuracy_xentropy = []
for line in f_tree1:
    line = line.strip()
    if len(line)>1:
        acc_start = line.find('Accuracy:')
        acc_end  = acc_start + len('Accuracy:')

        acc = float(line[acc_end:].strip())

        accuracy_tree.append(acc)
for line in f_tree2:
    line = line.strip()
    if len(line)>1:
        acc_start = line.find('Accuracy:')
        acc_end  = acc_start + len('Accuracy:')

        acc = float(line[acc_end:].strip())

        accuracy_tree.append(acc)
for line in f_tree3:
    line = line.strip()
    if len(line)>1:
        acc_start = line.find('Accuracy:')
        acc_end  = acc_start + len('Accuracy:')

        acc = float(line[acc_end:].strip())

        accuracy_tree.append(acc)
for line in f_tree4:
    line = line.strip()
    if len(line)>1:
        acc_start = line.find('Accuracy:')
        acc_end  = acc_start + len('Accuracy:')

        acc = float(line[acc_end:].strip())

        accuracy_tree.append(acc)
for line in f_tree5:
    line = line.strip()
    if len(line)>1:
        acc_start = line.find('Accuracy:')
        acc_end  = acc_start + len('Accuracy:')

        acc = float(line[acc_end:].strip())

        accuracy_tree.append(acc)
for line in f_xentropy:
    line = line.strip()
    if len(line)>1:
        acc_start = line.find('Accuracy:')
        acc_end  = acc_start + len('Accuracy:')

        acc = float(line[acc_end:].strip())

        accuracy_xentropy.append(acc)

classes = np.arange(5,1001)
plt.figure(0, figsize=(10,5))
l1, = plt.plot(classes, accuracy_tree)
l2, = plt.plot(classes, accuracy_xentropy)
plt.legend(handles=[l1,l2],labels=['Tree Loss', 'Cross Entropy Loss'], fontsize=15)
plt.xlabel('$d_(n=1000, k=100, d=1000)$', fontsize=15)
plt.ylabel('Accuracy', fontsize=15)
plt.savefig('accuracy_vs_d__.png', dpi=300)