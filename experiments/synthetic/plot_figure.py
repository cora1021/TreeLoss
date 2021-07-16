import matplotlib.pyplot as plt
import argparse
parser = argparse.ArgumentParser(description='create figures')
parser.add_argument('--experiment', choices=['loss_vs_n','loss_vs_d', 'loss_vs_sigma'], required=True)
args = parser.parse_args()

f = open(f'{args.experiment}_tree.txt', 'r')
f_ = open(f'{args.experiment}_xentropy.txt', 'r')
loss = []
W_err = []
accuracy = []
for line in f:
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

        loss.append(loss__)
        W_err.append(W_err__)
        accuracy.append(accuracy__)
loss_ = []
W_err_ = []
accuracy_ = []
for line in f_:
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

        loss_.append(loss__)
        W_err_.append(W_err__)
        accuracy_.append(accuracy__)

if args.experiment == 'loss_vs_n' :
    x = [16,32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    plt.figure(0)
    l1, = plt.plot(x, loss)
    l2, = plt.plot(x, loss_, linestyle="-.")
    plt.legend(handles=[l1,l2],labels=['Cover Tree Loss','Cross Entropy Loss'])
    plt.xlabel('Number of Data Points')
    plt.ylabel('Loss Value')
    plt.title('Loss vs Number of Data Points')
    plt.savefig('loss_vs_n.png', dpi=300)

    plt.figure(1)
    l1, = plt.plot(x, W_err)
    l2, = plt.plot(x, W_err_, linestyle="-.")
    plt.legend(handles=[l1,l2],labels=['Cover Tree Loss','Cross Entropy Loss'])
    plt.xlabel('Number of Data Points')
    plt.ylabel('Norm of Parameter Matrix Error')
    plt.title('Parameter Error vs Number of Data Points')
    plt.savefig('error_vs_n.png', dpi=300)

    plt.figure(2)
    l1, = plt.plot(x, accuracy)
    l2, = plt.plot(x, accuracy_, Linestyle="-.")
    plt.legend(handles=[l1,l2],labels=['Cover Tree Loss','Cross Entropy Loss'])
    plt.xlabel('Number of Data Points')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Number of Data Points')
    plt.savefig('accuracy_vs_n.png', dpi=300)


if args.experiment == 'loss_vs_d':
    x = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    plt.figure(0)
    l1, = plt.plot(x, loss)
    l2,= plt.plot(x, loss_, linestyle="-.")
    plt.legend(handles=[l1,l2],labels=['Cover Tree Loss','Cross Entropy Loss'])
    plt.xlabel('Dimension')
    plt.ylabel('Loss Value')
    plt.title('Loss vs Dimension')
    plt.savefig('loss_vs_d.png', dpi=300)

    plt.figure(1)
    l1, = plt.plot(x, W_err)
    l2, = plt.plot(x, W_err_, linestyle="-.")
    plt.legend(handles=[l1,l2],labels=['Cover Tree Loss','Cross Entropy Loss'])
    plt.xlabel('Dimension')
    plt.ylabel('Norm of Parameter Matrix Error')
    plt.title('Parameter Error vs Dimension')
    plt.savefig('error_vs_d.png', dpi=300)

    plt.figure(2)
    l1, = plt.plot(x, accuracy)
    l2, = plt.plot(x, accuracy_, Linestyle="-.")
    plt.legend(handles=[l1,l2],labels=['Cover Tree Loss','Cross Entropy Loss'])
    plt.xlabel('Number of Data Points')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Dimension')
    plt.savefig('accuracy_vs_d.png', dpi=300)

if args.experiment == 'loss_vs_sigma':
    x = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25]
    plt.figure(0)
    l1, = plt.plot(x, loss)
    l2, = plt.plot(x, loss_, linestyle="-.")
    plt.legend(handles=[l1,l2],labels=['Cover Tree Loss','Cross Entropy Loss'])
    plt.xlabel('Randomness')
    plt.ylabel('Loss Value')
    plt.title('Loss vs Randomness')
    plt.savefig('loss_vs_sigma.png', dpi=300)

    plt.figure(1)
    l1, = plt.plot(x, W_err)
    l2, = plt.plot(x, W_err_, Linestyle="-.")
    plt.legend(handles=[l1,l2],labels=['Cover Tree Loss','Cross Entropy Loss'])
    plt.xlabel('Randomness')
    plt.ylabel('Norm of Parameter Matrix Error')
    plt.title('Parameter Error vs Randomness')
    plt.savefig('error_vs_sigma.png', dpi=300)

    plt.figure(2)
    l1, = plt.plot(x, accuracy)
    l2, = plt.plot(x, accuracy_, Linestyle="-.")
    plt.legend(handles=[l1,l2],labels=['Cover Tree Loss','Cross Entropy Loss'])
    plt.xlabel('Randomness')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Randomness')
    plt.savefig('accuracy_vs_sigma.png', dpi=300)