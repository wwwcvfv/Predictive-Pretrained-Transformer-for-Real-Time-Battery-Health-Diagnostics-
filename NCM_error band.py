import pickle
import numpy as np
import matplotlib.pyplot as plt

def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

#
result = load_obj(r'./data/res_dict.pkl')
cycle_rmse = load_obj(r'./cycle_rmse_ncm_dict.pkl')

#
specified_batteries = [
    "CY35-05_1-#4.csv",
    "CY45-05_1-#24.csv",
    "CY45-05_1-#25.csv",
    "CY45-05_1-#27.csv"
]

custom_titles = [
    "NCM-1",
    "NCM-2",
    "NCM-3",
    "NCM-4"
]
#

plt.figure(figsize=(12, 8))  #
for i, (battery_name, custom_title) in enumerate(zip(specified_batteries, custom_titles), 1):
    soh_true = result[battery_name]['soh']['true']
    soh_pred = result[battery_name]['soh']['transfer']
    errors = cycle_rmse[battery_name]

    cycles = np.arange(len(soh_true))

    ax = plt.subplot(2, 2, i)  #
    ax.fill_between(cycles, soh_pred - errors, soh_pred + errors, color='#b5aad5', alpha=0.5, label='Error band')  #
    ax.scatter(cycles, soh_pred, color='#ab5399', label='Estimated capacity', s=6, zorder=3)  #
    ax.scatter(cycles, soh_true, color='#5392ba', label='Actual capacity', s=6, zorder=3)  #

    ax.set_title(custom_title, fontsize=16, fontname='Times New Roman')  #
    ax.set_xlabel('Cycle number', fontsize=16, fontname='Times New Roman')
    ax.set_ylabel('Estimated capacity (mAh)', fontsize=16, fontname='Times New Roman')
    ax.legend(fontsize=25, prop={'family': 'Times New Roman','size': 16})

    plt.xticks(size=16, family='Times New Roman')
    plt.yticks(size=16, family='Times New Roman')
plt.tight_layout()
save_path = r'./NCM_error band.tiff'
plt.savefig(save_path, dpi=300, format="tiff")
plt.show()