import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#
file_path = './data/plicy-1.xlsx'
data = pd.read_excel(file_path, sheet_name='plicy')

#
CC1 = data.iloc[:, 0]
CC2 = data.iloc[:, 1]

#
FS = 35

fig = plt.figure(figsize=(20, 10))

#
ax = sns.kdeplot(
    x=CC1,
    y=CC2,
    cmap="Blues",
    shade=True,
    bw_adjust=1.3,
    cbar=True
)

#
plt.tick_params(axis='x', labelsize=35)
plt.tick_params(axis='y', labelsize=35)

#
for label in ax.get_xticklabels():
    label.set_family('Times New Roman')
for label in ax.get_yticklabels():
    label.set_family('Times New Roman')
#
cbar = ax.figure.axes[-1]  #
cbar.tick_params(labelsize=25)
#
for l in cbar.get_yticklabels():
    l.set_family('Times New Roman')

#
plt.xlabel('CC 1 (C rate)', fontsize=FS, family='Times New Roman')
plt.ylabel('CC 2 (C rate)', fontsize=FS, family='Times New Roman')
plt.title('Constant current charging stage (0 to 80% SOC)', fontsize=FS, family='Times New Roman')
plt.show()
plt.savefig('charge_158.tiff', dpi=800, format="tiff")
