import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def get_info(fn):
    fn = os.path.splitext(fn)[0]
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
        ts[e] = 1-t
        vs[e] = 1-v

    for i in xrange(len(ts)):
        t_acc.append(ts[i])
        v_acc.append(vs[i])
    return t_acc, v_acc

folder = sys.argv[1]
folder2 = sys.argv[2]
output = sys.argv[3]

t_acc, v_acc = get_accuracies(folder)
t_acc2, v_acc2 = get_accuracies(folder2)

ts = t_acc[:4] + v_acc2[16:]
vs = v_acc

ts = ts[:100]
vs = vs[:100]
n = len(ts)

plt.title("Top 1 error rate")
t, = plt.plot(ts, 'r')
v, = plt.plot(vs, 'b')

plt.axis([0, n, 0, 1])
plt.xlabel("Epoch")
plt.ylabel("Error rate")
plt.legend((t,v), ('Training', 'Validation'))
plt.savefig(output)
#plt.show()



