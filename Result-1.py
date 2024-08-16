import pickle
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import numpy as np
import seaborn as sns
from scipy import interpolate
from datetime import datetime
import pandas as pd
import matplotlib.lines as mlines
import matplotlib.markers as mmarkers
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

result = load_obj('./result/res_dict')#
#
test = list(result.keys())
error_list = []

new_test = test
test_channel = [i for i in range(len(new_test))]
test_nm_ch = set(zip(test_channel,new_test))

sort_new_test = list(test_nm_ch)
sort_new_test = sorted(sort_new_test, key = lambda x: int(x[0]))

test_name = [i[0] for i in sort_new_test]
test_channel = [i[1] for i in sort_new_test]
sort_new_test =[(0, 'a5'), (1, 'a6'), (2, 'a7'), (3, 'a9'), (4, 'a11'), (5, 'a17')]
names=['16','30','7','29','28','21']
#
fig, ax = plt.subplots(figsize=(10, 9))
cmap = plt.get_cmap('YlGnBu')
x_lim = 2000
norm = plt.Normalize(vmin=0.3, vmax=1)
colors = ['#BE66A7', '#705183', '#7B8ECB', '#ABC979', '#F8B47C', '#DE7A66']
#
font_prop = FontProperties(family='Times New Roman', size=25)#
#
for i ,(code, name) in enumerate(sort_new_test):

        Soh_true = result[name]['soh']['true']
        Soh_pred = result[name]['soh']['transfer']
        interval = 15
        ax.plot(range(0, len(Soh_true), interval), Soh_true[::interval], '-', linewidth=4, c=colors[code])
        ax.plot(range(0, len(Soh_pred), interval), Soh_pred[::interval], '.', markersize=18, c=colors[code], label=f'Battery {names[i]}')
#
plt.legend(fontsize=20, loc="lower left",prop=font_prop)
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

plt.savefig('Figure/soh_1.tiff', dpi=800, format="tiff")
plt.show()

# density
fig = plt.figure(figsize=(10, 9))
error_list = []
for (code, name) in sort_new_test[:]:
    if code in range(0, 6):
        interval = 1
        rul_true = result[name]['soh']['true']
        rul_pred = result[name]['soh']['transfer']
        tmp = rul_true[::interval] - rul_pred[::interval]  #
        error_list.append(tmp.reshape(-1, 1))
error_array = np.vstack(error_list)
plt.xticks([-8, -2, 4], size=30, family='Times New Roman')
plt.xlim(-8.1, 4.5)
plt.yticks([], size=30, family='Times New Roman')
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.hist(error_array * 1000, bins=40, edgecolor='#FFA4B2', color='#F7CED5')  #
plt.title('Error Distribution', fontsize=35, family='Times New Roman')
plt.ylabel('Frequency', fontsize=35, family='Times New Roman')
plt.savefig('Figure/dsoh_1.tiff', dpi=800, format="tiff")
plt.show()
