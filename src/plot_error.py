import os
import sys
import matplotlib.pyplot as plt

def get_info(fn):
    fn = os.splitext(fn)[0]
    parts = fn.split('-')
    epoch = int(parts[0].split('.')[1])
    t = float(parts[1])
    v = float(parts[2])
    return epoch, t, v

def get_accuracies(folder):
    t_acc = []
    v_acc = []

    ts = {}
    vs = {}
    for fn in os.listdir(folder):
        e, t, v = get_info(fn)
        ts[e] = t
        vs[e] = v

    for i in xrange(len(t_accs)):
        t_acc.append(ts[i])
        v_acc.append(vs[i])
    return t_acc, v_acc

folder = sys.argv[1]
output = sys.argv[2]

t_acc, v_acc = get_accuracies(folder)
t_acc = range(10)
v_acc = range(0,20,2)
n = len(t_acc)

plt.title("Top 1 error rate")
t, = plt.plot(t_acc, 'r')
v, = plt.plot(v_acc, 'b')

plt.axis([0, n, 0, 1])
plt.xlabel("Epoch")
plt.ylabel("Error rate")
plt.legend((t,v), ('Training', 'Validation'))
plt.savefig(output)
plt.show()



