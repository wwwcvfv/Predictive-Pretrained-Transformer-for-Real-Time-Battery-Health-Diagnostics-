import pickle
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)

def load_obj(name):
    with open(name +'.pkl','rb') as f:
        return pickle.load(f)

result = load_obj('./result/res_dict(NCM)')

#
cycle_rmse_dict = {}

new_test = list(result.keys())
test_channel = ['#'+i[1:] for i in new_test]
test_nm_ch = set(zip(test_channel, new_test))
sort_new_test = list(test_nm_ch)

for (code, name) in sort_new_test:
    A = load_obj(f'D./data/NCM_DATA/{name}')[name]
    A_dq = A['dq'][1]
    soh_true = result[name]['soh']['true']
    soh_pred = result[name]['soh']['transfer']

    # SOH_true = soh_true / A_dq
    # SOH_pred = soh_pred / A_dq

    #
    rmse_per_cycle = np.sqrt((soh_pred - soh_true)**2)
    cycle_rmse_dict[name] = np.array(rmse_per_cycle)  #

#
for key, value in cycle_rmse_dict.items():
    print(f'{key}: {value[:5]}')  #

#
with open('cycle_rmse_ncm_dict.pkl', 'wb') as f:
    pickle.dump(cycle_rmse_dict, f)

#
with open('cycle_rmse_ncm_dict.pkl', 'rb') as f:
    loaded_rmse_dict = pickle.load(f)
    print(loaded_rmse_dict)
