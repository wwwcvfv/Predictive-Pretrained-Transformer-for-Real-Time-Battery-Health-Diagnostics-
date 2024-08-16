import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy import interpolate
import random
import pickle

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)  #


def change(arr, t, num):  #
    x_new = np.linspace(t[0], t[-1], num)  #
    f_linear = interpolate.interp1d(t, arr)  #
    y_new = f_linear(x_new)  #
    return y_new
#
path1 = './data/2017-05-12_batchdata_updated_struct_errorcorrect.mat'
path2 = './data/2017-06-30_batchdata_updated_struct_errorcorrect.mat'
path3 = './data/2018-04-12_batchdata_updated_struct_errorcorrect.mat'
path4 = './data/2018-02-20_batchdata_updated_struct_errorcorrect.mat'
path5 = './data/2019-01-24_batchdata_updated_struct_errorcorrect.mat'


temp1 = h5py.File(path1, 'r')
temp2 = h5py.File(path2, 'r')
temp3 = h5py.File(path3, 'r')
temp4 = h5py.File(path4, 'r')
temp5 = h5py.File(path5, 'r')

batch1 = temp1['batch']
batch2 = temp2['batch']
batch3 = temp3['batch']
batch4 = temp4['batch']
batch5 = temp5['batch']


#
cycle_life = dict()
temp = temp1
batch = batch1
for bat_num in range(batch['cycles'].shape[0]):
    a = list(temp[batch['cycle_life'][bat_num, 0]])[0][0]
    if np.isnan(a):
        continue
    else:
        a = int(a)
    #
    cycle_life.update({'a' + str(bat_num): a})

temp = temp2
batch = batch2
for bat_num in range(batch['cycles'].shape[0]):
    a = list(temp[batch['cycle_life'][bat_num, 0]])[0][0]
    if np.isnan(a):
        continue
    else:
        a = int(a)
    cycle_life.update({'b' + str(bat_num): a})

temp = temp3
batch = batch3
for bat_num in range(batch['cycles'].shape[0]):
    a = list(temp[batch['cycle_life'][bat_num, 0]])[0][0]
    if np.isnan(a):
        continue
    else:
        a = int(a)
    cycle_life.update({'c' + str(bat_num): a})
cycle_life.update({'c23': 2190})  # 23 32
cycle_life.update({'c32': 2238})

temp = temp4
batch = batch4
for bat_num in range(batch['cycles'].shape[0]):
    a = list(temp[batch['cycle_life'][bat_num, 0]])[0][0]
    if np.isnan(a):
        continue
    else:
        a = int(a)
    cycle_life.update({'d' + str(bat_num): a})

temp = temp5
batch = batch5
for bat_num in range(batch['cycles'].shape[0]):
    a = list(temp[batch['cycle_life'][bat_num, 0]])[0][0]
    if np.isnan(a):
        continue
    else:
        a = int(a)
    cycle_life.update({'e' + str(bat_num): a})
print('cycle_life:', len(cycle_life))       #
##
#
del cycle_life['a0']
del cycle_life['a1']
del cycle_life['a2']
del cycle_life['a3']#
del cycle_life['a4']#
# remove batteries that do not reach 80% capacity
del cycle_life['a8']
del cycle_life['a10']
del cycle_life['a12']
del cycle_life['a13']
del cycle_life['a22']

# data incomplete
del cycle_life['b7']
del cycle_life['b8']
del cycle_life['b9']
del cycle_life['b15']
del cycle_life['b16']

# remove noisy channels from c
del cycle_life['c37']
del cycle_life['c2']
del cycle_life['c23']
del cycle_life['c32']
del cycle_life['c38']
del cycle_life['c39']

cycle_test = []
cycle_else = []
for key in cycle_life:
    if 'a' in key:
        cycle_test.append(key)
    else:
        cycle_else.append(key)

#
random.seed(1002)

cycle_new = []
for key in cycle_test:
    if key in cycle_life:
        cycle_new.append(key)
print(cycle_new)

cmap = plt.cm.viridis#
fig, ax = plt.subplots(figsize=(20, 10))  #

Cycle = []
DQ = []
n_cyc = 100
for i in cycle_new:
    key = i
    bat_life = cycle_life[key]
    if bat_life < n_cyc:  #
        continue
    if key[0] == 'a':
        temp = temp1
        batch = batch1
    elif key[0] == 'b':
        temp = temp2
        batch = batch2
    elif key[0] == 'c':
        temp = temp3
        batch = batch3
    elif key[0] == 'd':
        temp = temp4
        batch = batch4
    elif key[0] == 'e':
        temp = temp5
        batch = batch5

    bat_num = int(key[1:])
    if False:
        pass

    else:

        Qd = list(temp[batch['summary'][bat_num, 0]]['QDischarge'][0, :])
        cycle = list(temp[batch['summary'][bat_num, 0]]['cycle'][0,:]) # 获取引用
        DQ.append(Qd)
        Cycle.append(cycle)
        print(f"Total number of sequences in Cycle: {len(Cycle)}")

max_num = []
for idx,cycle_nm in enumerate(Cycle):
    max_num.append(np.max(cycle_nm))
# Normalize cycle life for color mapping
    norm = plt.Normalize(min(max_num), max(max_num))

for i, (q, cycle_num) in enumerate(zip(DQ, Cycle)):
    cycle_n = np.max(cycle_num)
    color = cmap(norm(cycle_n))
    plt.plot(list(q), color=color, linewidth=2)#[::2]


plt.ylim(0.88, 1.1)  #
plt.xlim(0, 1800)
#
plt.tick_params(axis='x', labelsize=35)  #
plt.tick_params(axis='y', labelsize=35)  #
#
for label in ax.get_xticklabels():
    label.set_family('Times New Roman')

for label in ax.get_yticklabels():
    label.set_family('Times New Roman')

plt.xlabel('Cycle Number', fontsize=35, family='Times New Roman')
plt.ylabel('Discharge Capacity (Ah)', fontsize=35, family='Times New Roman')

#
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  #

#
cbar = fig.colorbar(sm, ax=ax)  #
cbar.set_label('Cycle Number', fontsize=35, family='Times New Roman')
cbar.ax.tick_params(labelsize=25)  #
#
for l in cbar.ax.yaxis.get_ticklabels():
    l.set_family('Times New Roman')

plt.savefig('Figure/cycle_life.tiff', dpi=800, format="tiff")
plt.show()
