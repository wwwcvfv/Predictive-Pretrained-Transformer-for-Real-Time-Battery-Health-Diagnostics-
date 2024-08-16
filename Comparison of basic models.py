import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.collections import PolyCollection

#
file_path1 = './data/cnn.xlsx'
file_path2 = './data/lstm.xlsx'
file_path3 = './data/transformer.xlsx'
file_path4 = './data/PPT.xlsx'

data1 = pd.read_excel(file_path1)
data2 = pd.read_excel(file_path2)
data3 = pd.read_excel(file_path3)
data4 = pd.read_excel(file_path4)

# Metric names
metrics = ['rmse_after', 'r2_after', 'wmape_after','mae_after']
# Labels for the legend
file_labels = ['RMSE ', 'R\u00b2 ', 'MAE ', 'WMAPE ']
# Iterate through each metric and plot
for metric_index, metric in enumerate(metrics):
    plt.figure(figsize=(12, 9))

    #
    data = {
        'A': data1[metric][:36],
        'B': data2[metric][:36],
        'C': data3[metric][:36],
        'D': data4[metric][:36],
    }
    df = pd.DataFrame(data)

    df_melt = pd.melt(df, var_name='Group', value_name='Value')

    palette = {
        'A': '#E4C0DB',
        'B': '#81BB82',
        'C': '#ED91A0',
        'D': '#06BFE4',
    }

    sns.violinplot(x='Group', y='Value', data=df_melt, inner=None, palette=palette, width=0.5, linewidth=2)

    ax = plt.gca()
    for art in ax.findobj(match=PolyCollection):
        for path in art.get_paths():
            vertices = path.vertices
            center = np.median(vertices[:, 0])
            vertices[:, 0] = np.maximum(vertices[:, 0], center)

    #
    for pc in ax.collections:
        if isinstance(pc, PolyCollection):
            face_color = pc.get_facecolor()
            pc.set_edgecolor(face_color)

    positions = np.arange(len(data.keys())) - 0.12
    box_data = [df_melt[df_melt['Group'] == group]['Value'] for group in data.keys()]
    boxplot = ax.boxplot(box_data, positions=positions, widths=0.15, patch_artist=True, showfliers=False)

    #
    for i, box in enumerate(boxplot['boxes']):
        group_label = df_melt['Group'].unique()[i]
        color = palette[group_label]
        box.set_facecolor(color)
        box.set_edgecolor(color)
        box.set_linewidth(8)
        box.set_alpha(0.6)

        whiskers = boxplot['whiskers'][i * 2:(i + 1) * 2]
        caps = boxplot['caps'][i * 2:(i + 1) * 2]
        median = boxplot['medians'][i]
        for whisker in whiskers:
            whisker.set_color(color)
            whisker.set_linewidth(4)
        for cap in caps:
            cap.set_color(color)
            cap.set_linewidth(4)
        median.set_color(color)
        median.set_linewidth(4)

    #
    for i, group in enumerate(data.keys()):
        group_label = df_melt['Group'].unique()[i]
        color = palette[group_label]
        group_data = df_melt[df_melt['Group'] == group]['Value']
        x = np.random.normal(i - 0.33, 0.04, size=len(group_data))
        plt.scatter(x, group_data, color=color, alpha=0.8, edgecolor='none', s=100)

    #
    ax.set_xlim(-0.5, len(data) - 0.5)
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels(['CNN', 'LSTM', 'Transformer', 'Proposed\n method'])
    ax.tick_params(axis='x', labelsize=35)
    ax.tick_params(axis='y', labelsize=35)

    for label in ax.get_xticklabels():
        label.set_family('Times New Roman')
    for label in ax.get_yticklabels():
        label.set_family('Times New Roman')

    ax.set_xlabel('', fontsize=35, family='Times New Roman')
    ax.set_ylabel('', fontsize=35, family='Times New Roman')
    #
    ax.set_title(file_labels[metric_index], family='Times New Roman', fontsize=35)
    #
    plt.savefig(f'Figure/{metric}.tiff', dpi=800, format="tiff")
    plt.show()
