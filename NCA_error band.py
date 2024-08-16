import pickle
import numpy as np
import matplotlib.pyplot as plt

def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

#
result = load_obj(r'./data/res_dict.pkl')
cycle_rmse = load_obj(r'./cycle_rmse_nca_dict.pkl')

#
specified_batteries = [
    ("CY45-05_1-#26.csv", None),
    ("CY45-05_1-#21.csv", None),
    ("CY45-05_1-#22.csv", None),
    ("CY45-05_1-#23.csv", None),  #
]
custom_titles = [
    "NCA-1",
    "NCA-2",
    "NCA-3",
    "NCA-4"
]
plt.figure(figsize=(12, 8))  #

#
for i, (battery_name, max_cycles) in enumerate(specified_batteries, 1):
    soh_true = result[battery_name]['soh']['true']
    soh_pred = result[battery_name]['soh']['transfer']
    errors = cycle_rmse[battery_name]

    if max_cycles is not None:
        soh_true = soh_true[:max_cycles]
        soh_pred = soh_pred[:max_cycles]
        errors = errors[:max_cycles]
        cycles = np.arange(max_cycles)
    else:
        cycles = np.arange(len(soh_true))

    ax = plt.subplot(2, 2, i)
    ax.fill_between(cycles, soh_pred - errors, soh_pred + errors, color='#fb9a9b', alpha=0.4,label='Error band')  #
    ax.scatter(cycles, soh_pred, color='#EC8D9C', label='Estimated capacity', s=6, zorder=3)
    ax.scatter(cycles, soh_true, color='#767171', label='Actual capacity', s=6, zorder=3)  #

    ax.set_title(custom_titles[i-1], fontsize=16, fontname='Times New Roman')
    ax.set_xlabel('Cycle number', fontsize=16, fontname='Times New Roman')
    ax.set_ylabel('Estimated capacity (mAh)', fontsize=16, fontname='Times New Roman')
    ax.legend(fontsize=25, prop={'family': 'Times New Roman','size': 16})
    plt.xticks(size=16, family='Times New Roman')
    plt.yticks(size=16, family='Times New Roman')
plt.tight_layout()
save_path = r'Figure/NCA_error band.tiff'
plt.savefig(save_path, dpi=800, format="tiff")
plt.show()
