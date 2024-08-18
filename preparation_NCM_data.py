import pandas as pd
import numpy as np
import os
import pickle
import tqdm

from scipy import interpolate



import warnings
warnings.filterwarnings("ignore")
# save dict
def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)


# load dict
def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def interp(v, q, num):
    f = interpolate.interp1d(v, q, kind='linear')
    v_new = np.linspace(v[0], v[-1], num)
    q_new = f(v_new)
    vq_new = np.concatenate((v_new.reshape(-1, 1), q_new.reshape(-1, 1)), axis=1)
    return q_new

data_path = './data/Dataset_2_NCM_battery/'  #
files = os.listdir(data_path)  #
files = [file for file in files if any(substring in file for substring in ['CY25-05_1', 'CY35-05_1', 'CY45-05_1'])]
bat_prefix = list(set(files))  #

bat_prefix = tqdm.tqdm(bat_prefix)  
for prefix in bat_prefix:
    cyc_v = {}
    cyc_rul = {}
    cyc_dq = {}
    #A = {}

    cycle_df = pd.read_csv(os.path.join(data_path, prefix)) 
    tmp = cycle_df[['cycle number', 'Q discharge/mA.h']]

    cycle_life = int(tmp['cycle number'].iloc[-1]) 
    for j in range(1, cycle_life + 1):

        A = tmp[tmp['cycle number'] == j]['Q discharge/mA.h'].iloc[-1]
        if A < 1500:
            continue
        else:
            cyc_rul[j] = cycle_life - j
            cyc_dq[j] = A
            tmp_cyc = cycle_df[cycle_df['cycle number'] == j]  #
            tmp_cyc = tmp_cyc.reset_index(drop=True)  #
            cyc_v[j] = tmp_cyc  #

    #
    bats_dic = {}
    bats_dic[prefix] = {'rul': cyc_rul,
                        'dq': cyc_dq,
                        'data': cyc_v}
    save_obj(bats_dic, './data/NCM_DATA/' + prefix)  #

pkl_list = os.listdir('./data/NCM_DATA/')
#
target_strings = ['CY25-05_1', 'CY35-05_1', 'CY45-05_1']
pkl_list = [i for i in pkl_list if any(target in i for target in target_strings)]


train_name = []
for name in pkl_list:
    train_name.append(name[:-4])


def get_xy(name):
    A = load_obj(f'./data/NCM_DATA/{name}')[name]  #
    A_rul = A['rul']
    A_dq = A['dq']
    A_df = A['data']
    all_idx = list(A_dq.keys())[9:]  #
    all_fea, all_lbl, aux_lbl = [], [], []
    for cyc in all_idx:
        tmp = A_df[cyc]
        posi = (tmp['Q discharge/mA.h']>0).argmax()-1#
        init_cap =tmp['Q charge/mA.h'].iloc[posi] * 0.8  #
        left = (tmp['Q charge/mA.h'] > init_cap).argmax() - 20
        current = tmp['<I>/mA'].values
        for i in range(len(current)):#
            if current[i] > 0:  #
                break
        i += 1
        pos = np.where(current < current[i])[0]  #
        for j in pos:
            if j > i:
                break
        right = j + 20
        if left >= right - 1:
            continue

        tmp = tmp.iloc[left:right]

        tmp_v = tmp['Ecell/V'].values
        tmp_q = tmp['Q charge/mA.h'].values
        tmp_t = tmp['time/s'].values
        v_fea = interp(tmp_t, tmp_v, fea_num)
        q_fea = interp(tmp_t, tmp_q, fea_num)

        tmp_fea = np.hstack((v_fea.reshape(-1, 1), q_fea.reshape(-1, 1)))

        all_fea.append(np.expand_dims(tmp_fea, axis=0))
        all_lbl.append(A_rul[cyc])  #
        aux_lbl.append(A_dq[cyc])
    #
    all_fea = np.vstack(all_fea)
    all_lbl = np.array(all_lbl)
    aux_lbl = np.array(aux_lbl)

    all_fea_c = all_fea.copy()
    all_fea_c[:, :, 0] = (all_fea_c[:, :, 0] - v_low) / (v_upp - v_low)  #
    all_fea_c[:, :, 1] = (all_fea_c[:, :, 1] - q_low) / (q_upp - q_low)
    dif_fea = all_fea_c - all_fea_c[0:1, :, :]
    all_fea = np.concatenate((all_fea, dif_fea), axis=2)

    all_fea = np.lib.stride_tricks.sliding_window_view(all_fea, (n_cyc, fea_num, 4))  #
    aux_lbl = np.lib.stride_tricks.sliding_window_view(aux_lbl, (n_cyc,))

    all_fea = all_fea.squeeze(axis=(1, 2,))

    all_lbl = all_lbl[n_cyc - 1:]  #
    all_fea = all_fea[::stride]
    all_fea = all_fea[:, ::in_stride, :, :]
    all_lbl = all_lbl[::stride]
    aux_lbl = aux_lbl[::stride]
    aux_lbl = aux_lbl[:, ::in_stride, ]

    all_fea_new = np.zeros(all_fea.shape)
    all_fea_new[:, :, :, 0] = (all_fea[:, :, :, 0] - v_low) / (v_upp - v_low)  #
    all_fea_new[:, :, :, 1] = (all_fea[:, :, :, 1] - q_low) / (q_upp - q_low)
    all_fea_new[:, :, :, 2] = all_fea[:, :, :, 2]  #
    all_fea_new[:, :, :, 3] = all_fea[:, :, :, 3]
    print(f'{name} length is {all_fea_new.shape[0]}',
          'v_max:', '%.4f' % all_fea_new[:, :, :, 0].max(),
          'q_max:', '%.4f' % all_fea_new[:, :, :, 1].max(),
          'dv_max:', '%.4f' % all_fea_new[:, :, :, 2].max(),
          'dq_max:', '%.4f' % all_fea_new[:, :, :, 3].max())
    all_lbl = all_lbl / lbl_factor
    aux_lbl = aux_lbl / aux_factor

    return all_fea_new, np.hstack((all_lbl.reshape(-1, 1), aux_lbl))


n_cyc = 30
in_stride = 3
fea_num = 100

v_low = 3.5
v_upp = 4.3
q_low = 2200
q_upp = 3500
lbl_factor = 2200
aux_factor = 3500

stride = 1
all_loader = dict()
all_fea = []
all_lbl = []
print('----init_train----')
if os.path.exists('./data/NCM_DATA/ncm_loader.pkl'):

    all_loader = load_obj('./data/NCM_DATA/ncm_loader')
else:

    for name in train_name:  #
        tmp_fea, tmp_lbl = get_xy(name)
        all_loader.update({name: {'fea': tmp_fea, 'lbl': tmp_lbl}})
        all_fea.append(tmp_fea)
        all_lbl.append(tmp_lbl)
    save_obj(all_loader, './data/NCM_DATA/ncm_loader')
