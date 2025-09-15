import torch
import json
import sys
import numpy as np
import pymap3d as p3d
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import pickle
import tasgnss as tas

SYS_MAP = {'G':0,'R':1,'E':2,'C':3,'J':4}



def one_hot(s_sys):
    indices = np.array([SYS_MAP[char] for char in s_sys])
    indices_tensor = torch.tensor(indices)
    one_hot_encoded = F.one_hot(indices_tensor, num_classes=len(SYS_MAP))
    return one_hot_encoded


try:
    config = sys.argv[1]
except:
    config = "config/test_klt1.json"

try:
    model_file = sys.argv[2]
except:
    model_file = None

with open(config) as f:
    conf = json.load(f)

mode = conf['mode']
if mode not in ['train','predict']:
    raise RuntimeError("%s is not a valid option"%mode)

result_path = conf['save_path']
result_model_path = result_path+"/model"
os.makedirs(result_model_path,exist_ok=True)
DEVICE = conf.get('device', 'cpu')

try:
    with open(result_path+"/preprocess.pkl",'rb') as f:
        sols = pickle.load(f)
except FileNotFoundError:
    raise RuntimeError("No preprocess.pkl found, please first run preprocess.py")


pos_errs = []
res = []

with torch.no_grad():
    with tqdm(range(len(sols))) as t:
        for i in t:
            sol = sols[i]
            sol_id = id(sol)
            gnss = sol['gnss']
            p = gnss['pos']
            p_t = gnss['cb']
            data = gnss['data']
            s_sys = data[:,2]
            SNR = data[:,9].astype(np.float64)
            residual = gnss['residual_info']['residual'].squeeze()
            el = data[:,12].astype(np.float64)
            az = data[:,11].astype(np.float64)
            if np.linalg.norm(residual) > 500:
                continue
            if SNR.shape[0] != el.shape[0] or SNR.shape[0] != residual.shape[0]:
                print("shape mismatch")
                continue
            tas.cache_data[sol_id] = [p, p_t, None, None, gnss['data'], gnss['solve_data'], gnss['raw_data']]

            SNR = torch.tensor(SNR, dtype=torch.float64).to(DEVICE).reshape(-1,1)
            el = torch.tensor(el, dtype=torch.float64).to(DEVICE).reshape(-1,1)

            in_data = torch.hstack([SNR,el])

            weight = torch.tensor(tas.goGPSW(in_data))
            
            pos = tas.wls_pnt_pos(sol,None,use_cache=True,w = weight, enable_torch=True, device=DEVICE)['pos']
            gt = sol['gt']
            enu = torch.hstack(p3d.ecef2enu(*pos[:3], gt[1], gt[2], gt[3]))+torch.tensor(gt[7:10]).to(DEVICE)
            err_2d = torch.norm(enu[:2])
            err_3d = torch.norm(enu)
            t.set_description(f"2D error: {err_2d:.2f}, 3D error: {err_3d:.2f}")
            pos_errs.append([err_2d.cpu().numpy(),err_3d.cpu().numpy()])
            res.append(pos[:3].cpu().numpy())

    

pos_errs = np.array(pos_errs)
plt.plot(pos_errs[:,0])
plt.savefig(result_path + "/pos_errs_2d_goGPS.png")
plt.clf()
plt.plot(pos_errs[:,1])
plt.savefig(result_path + "/pos_errs_3d_goGPS.png")
np.savetxt(result_path + "/pos_errs_goGPS.csv", pos_errs.reshape(-1,2), delimiter=',')
res = np.array(res)
np.savetxt(result_path + "/pos_goGPS.csv", res, delimiter=',')
print(pos_errs.mean(axis=0))