import matplotlib.pyplot as plt
import numpy as np
import math

f = open('norm.txt', 'r')

W = []
V = []
for line in f:
    line = line.strip()
    if len(line)>1:
        W_start = line.find('W_norm:')
        W_end  = W_start + len('W_norm:')

        V_start = line.find('V_norm:')
        V_end  = V_start + len('V_norm:')

        w = float(line[W_end:V_start].strip())
        v = float(line[V_end:].strip())

        W.append(w)
        V.append(v)

classes = np.arange(10,1001)
plt.figure(0, figsize=(10,5))
l1, = plt.plot(classes, W)
l2, = plt.plot(classes, V)
plt.legend(handles=[l1,l2],labels=[r'$|{W}|$', r'$|{V}|$'], fontsize=15)
plt.xlabel('Number of Classes($k$)', fontsize=15)
plt.ylabel('Norm of Vector', fontsize=15)
plt.savefig('class_v_norm.png', dpi=300)