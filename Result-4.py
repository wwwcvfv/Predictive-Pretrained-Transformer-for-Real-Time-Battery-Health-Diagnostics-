import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

from matplotlib.font_manager import FontProperties
import h5py

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def change(arr, t, num):
    x_new = np.linspace(t[0], t[-1], num)
    f_linear = interpolate.interp1d(t, arr)
    y_new = f_linear(x_new)
    return y_new

#
path1 = './data/2017-05-12_batchdata_updated_struct_errorcorrect.mat'
temp1 = h5py.File(path1, 'r')
batch1 = temp1['batch']

# .
result = load_obj('./result/res_dict')
#
sort_new_test =[(18, 'a36'), (19, 'a37'), (20, 'a38'), (21, 'a43'), (22, 'a21'), (23, 'a44')]
names=['5','36','11','22','24','26']
#
fig, ax = plt.subplots(figsize=(10, 9))
cmap = plt.get_cmap('YlGnBu')
x_lim = 2000
norm = plt.Normalize(vmin=0.3, vmax=1)
colors = ['#9EA1A4','#A7917D','#81BB82','#ED91A0','#06BFE4' ,'#A071D9','#7B8ECB' ]
#
font_prop = FontProperties(family='Times New Roman', size=55)
marker_size = 10
#
font_prop = FontProperties(family='Times New Roman', size=25)#
for i ,(code, name) in enumerate(sort_new_test):
    Soh_true = result[name]['soh']['true']
    Soh_base = result[name]['soh']['base']
    Soh_pred = result[name]['soh']['transfer']

    interval = 15  #
    ax.plot(range(0, len(Soh_true), interval), Soh_true[::interval],'-', linewidth=4, c=colors[i])  #
    ax.plot(range(0, len(Soh_pred), interval), Soh_pred[::interval], '.', markersize=18, c=colors[i], label=f'Battery {names[i]}')
#
plt.legend(fontsize=20, loc="upper right",prop=font_prop)
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.tick_params(length=10)
plt.xticks([0, 500, 1000], size=25)
plt.yticks([0.9, 1.0, 1.1], size=25)
#
for label in ax.get_xticklabels():
    label.set_family('Times New Roman')
for label in ax.get_yticklabels():
    label.set_family('Times New Roman')

plt.ylabel('Estimated capacity (Ah)', fontsize=35, family='Times New Roman')
plt.xlabel('Cycles', fontsize=35, family='Times New Roman')
plt.savefig('Figure/soh_4.tiff', dpi=800, format="tiff")
plt.show()

#
# density
fig = plt.figure(figsize=(10, 9))
error_list = []
for i ,(code, name) in enumerate(sort_new_test):
    interval = 1
    rul_true = result[name]['soh']['true']
    rul_base = result[name]['soh']['base']
    rul_pred = result[name]['soh']['transfer']
    tmp = rul_true[::interval] - rul_pred[::interval]
    error_list.append(tmp.reshape(-1, 1))

error_array = np.vstack(error_list)
plt.xticks([-8, -2, 4], size=30, family='Times New Roman')
plt.xlim(-8.1, 4.5)
plt.yticks([], size=30, family='Times New Roman')
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.hist(error_array * 1000, bins=40, edgecolor='#FFA4B2', color='#F7CED5')
plt.title('Error Distribution', fontsize=35, family='Times New Roman')

plt.ylabel('Frequency', fontsize=35, family='Times New Roman')
# plt.grid(True)
# plt.show()
plt.savefig('Figure/dsoh_4.tiff', dpi=800, format="tiff")
plt.show()
