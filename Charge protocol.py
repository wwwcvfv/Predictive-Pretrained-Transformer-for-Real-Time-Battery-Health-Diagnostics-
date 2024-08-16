#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 14:02:51 2018

@author: peter
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams


fig, ax1 = plt.subplots(figsize=(20, 10))

FS = 35
LW = 3

rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
rcParams['font.size'] = FS
rcParams['axes.labelsize'] = FS
rcParams['xtick.labelsize'] = FS
rcParams['ytick.labelsize'] = FS
rcParams['font.sans-serif'] = ['Times New Roman']
rcParams['font.family'] = 'sans-serif'

########## 2a ##########

# Initialize axis limits
ax1.set_xlabel('State of charge (%)', fontsize=FS, family='Times New Roman')
ax1.set_ylabel('Current (C rate)', fontsize=FS, family='Times New Roman')
ax1.tick_params(axis='both', which='major', labelsize=FS, labelcolor='black')
ax1.set_xlim([0, 100])
ax1.set_ylim([0, 10])

#
ax1.set_xticks(np.arange(0, 101, 20))
ax1.set_xticklabels(np.arange(0, 101, 20), fontsize=FS, family='Times New Roman')
ax1.set_yticks(np.arange(0, 11, 2))
ax1.set_yticklabels(np.arange(0, 11, 2), fontsize=FS, family='Times New Roman')

# Add grey lines
C1list = [3.6, 4.4, 4.8, 5.2, 6.0, 7.0, 8.0]
C2list = [4.4, 4.8, 5.2, 5.6, 6.0, 7.0]
C3list = [4.4, 4.8, 5.2, 5.6]

for c1 in C1list:
    ax1.plot([0, 20], [c1, c1], linewidth=2, color='grey')
for c2 in C2list:
    ax1.plot([20, 40], [c2, c2], linewidth=2, color='grey')
for c3 in C3list:
    ax1.plot([40, 60], [c3, c3], linewidth=2, color='grey')

# Add example policy
c1, c2, c3, c4 = 7.0, 4.8, 4.8, 3.652
ax1.plot([0, 20], [c1, c1], linewidth=LW, color='#74CAA9')
ax1.plot([20, 40], [c2, c2], linewidth=LW, color='#74CAA9')
ax1.plot([40, 60], [c3, c3], linewidth=LW, color='#74CAA9')
ax1.plot([60, 80], [c4, c4], linewidth=LW, color='#60B3D4')

# Add bands
ax1.axvspan(0, 20, ymin=0.36, ymax=0.8, color='#74CAA9', alpha=0.25)
ax1.axvspan(20, 40, ymin=0.36, ymax=0.7, color='#74CAA9', alpha=0.25)
ax1.axvspan(40, 60, ymin=0.36, ymax=0.56, color='#74CAA9', alpha=0.25)
ax1.axvspan(60, 80, ymin=0, ymax=0.48, color='#60B3D4', alpha=0.25)

# Dotted lines for SOC bands
for k in [2, 4, 6, 8]:
    ax1.plot([k * 10, k * 10], [0, 10], linewidth=2, color='grey', linestyle=':')

# CC labels
label_height = 9.2
for k in np.arange(4):
    ax1.text(10 + 20 * k, label_height, 'CC' + str(k + 1), horizontalalignment='center', fontsize=FS, family='Times New Roman')
ax1.text(90, label_height, 'CC5-CV1', horizontalalignment='center', fontsize=FS, family='Times New Roman')

# Add 1C charging
ax1.plot([80, 89], [1, 1], linewidth=LW, color='#085A9E')
x = np.linspace(89, 100, 100)
y = np.exp(-0.5 * (x - 89))
ax1.plot(x, y, linewidth=LW, color='#085A9E')

# Charging time text box
ct_label_height = 0.5
ax1.plot([0.1, 0.1], [ct_label_height - 0.25, ct_label_height + 0.25], linewidth=3, color='grey')
ax1.plot([80, 80], [ct_label_height - 0.25, ct_label_height + 0.25], linewidth=2, color='grey')
ax1.plot([0, 80], [ct_label_height, ct_label_height], linewidth=2, color='grey')

textstr = 'Charging time (0 to 80% SOC) 10 minutes'
props = dict(boxstyle='round', facecolor='white', edgecolor='grey', alpha=1, linewidth=2)
ax1.text(0.4, ct_label_height / 10, textstr, transform=ax1.transAxes, fontsize=FS,
        verticalalignment='center', horizontalalignment='center', bbox=props)

# Voltage label text box
v_label_height = 8.4
v_label_lines = False
if v_label_lines:
    ax1.plot([0.1, 0.1], [v_label_height - 0.25, v_label_height + 0.25], linewidth=3, color='grey')
    ax1.plot([99.9, 99.9], [v_label_height - 0.25, v_label_height + 0.25], linewidth=3, color='grey')
    ax1.plot([0, 100], [v_label_height, v_label_height], linewidth=2, color='grey')

textstr = 'Max voltage = 3.6 V'
props = dict(boxstyle='round', facecolor='white', edgecolor='grey', alpha=1, linewidth=2)
ax1.text(0.5, v_label_height / 10, textstr, transform=ax1.transAxes, fontsize=FS,
         verticalalignment='center', horizontalalignment='center', bbox=props)

plt.savefig('charge_45.tiff', dpi=800, format="tiff")
plt.show()