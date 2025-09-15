from datetime import datetime
import torch
import json
import sys
import numpy as np
import pandas as pd
import pymap3d as p3d
from model.model import HybridShareSysNet
from torch.nn import HuberLoss,MSELoss
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import pickle
import tasgnss as tas

SYS_MAP = {'G':0,'R':1,'E':2,'C':3,'J':4}

global cache_ls
global cache_input
cache_ls = {}
cache_input = {}


def one_hot(s_sys):
    indices = np.array([SYS_MAP[char] for char in s_sys])
    indices_tensor = torch.tensor(indices)
    one_hot_encoded = F.one_hot(indices_tensor, num_classes=len(SYS_MAP))
    return one_hot_encoded


ctime = datetime.now().strftime("%Y-%m-%d_%H_%M")

try:
    config = sys.argv[1]
except:
    config = "config/train.json"

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
        pres = pickle.load(f)
except FileNotFoundError:
    raise RuntimeError("No preprocess.pkl found, please first run preprocess.py")



SNRS = []
residuals = []
els = []
azs = []

for sol in pres:
    sol_id = id(sol)
    gnss = sol['gnss']
    p = gnss['pos']
    p_t = gnss['cb']
    data = gnss['data']
    s_sys = data[:,2]
    SNR = data[:,9]
    residual = gnss['residual_info']['residual'].squeeze()
    el = data[:,12].astype(np.float64)
    az = data[:,11].astype(np.float64)
    if np.linalg.norm(residual) > 500:
        continue
    if SNR.shape[0] != el.shape[0] or SNR.shape[0] != residual.shape[0]:
        print("shape mismatch")
        continue
    tas.cache_data[sol_id] = [p, p_t, None, None, gnss['data'], gnss['solve_data'], gnss['raw_data']]
    # use cache to avoid duplicated loading
    cache_input[sol_id] = {
        'sys_oh': one_hot(s_sys).double().cpu(),
        'SNR': torch.tensor(SNR.astype(np.int8), dtype=torch.float64).cpu(),
        'residual': torch.tensor(residual, dtype=torch.float64).cpu(),
        'el': torch.tensor(el, dtype=torch.float64).cpu(),
        'az': torch.tensor(az, dtype = torch.float64).cpu(),
    }

    SNRS.append(SNR)
    residuals.append(residual)
    els.append(el)
    azs.append(az)

    
SNRS = np.hstack(SNRS)
residuals = np.hstack(residuals)
els = np.hstack(els)
azs = np.hstack(azs)
SNR_mean = SNRS.mean()
SNR_std = SNRS.std()
residual_mean = residuals.mean()
residual_std = residuals.std()
el_mean = els.mean()
el_std = els.std()
az_mean = azs.mean()
az_std = azs.std()
imean = np.array([SNR_mean,el_mean,az_mean,residual_mean,0,0,0,0,0])
istd = np.array([SNR_std,el_std,az_std,residual_std,1,1,1,1,1])

print(f"preprocess done, mean:{imean}, std:{istd}")


net = HybridShareSysNet(torch.tensor(imean,dtype=torch.float32),torch.tensor(istd,dtype=torch.float32))
net.double()
net = net.to(DEVICE)

pos_errs = []


opt = torch.optim.Adam(net.parameters(), lr=0.01)
epoch = conf.get('epoch', 120)
batch_size = conf.get('batch', 3000) 
lossFn = MSELoss(reduction='sum')
vis_loss = []

checkpoint_interval = 10  # 每40次迭代保存一次
checkpoint_dir = conf['save_path'] + "/model/"+ctime
os.makedirs(checkpoint_dir, exist_ok=True)
loss_type = conf['loss_type']
if loss_type == "2d":
    loss_locate = 2
else:
    loss_locate = 3



for k in range(epoch):
    total_loss = 0
    accumulated_loss = 0
    step = 0
    opt.zero_grad()  
    loss = 0
    total_len = len(pres)
    # shuffule the sols list
    np.random.shuffle(pres)
    with tqdm(range(len(pres)), desc=f"Epoch {k + 1}") as t:
        for i in t:
            sol = pres[i]
            if id(sol) not in cache_input:
                print("no cache for this epoch")
                continue

            pre_data = cache_input[id(sol)]

            gnss_data = torch.hstack([
                pre_data['SNR'].reshape(-1, 1).to(DEVICE),
                pre_data['el'].reshape(-1, 1).to(DEVICE),
                pre_data['az'].reshape(-1, 1).to(DEVICE),
                pre_data['residual'].reshape(-1, 1).to(DEVICE),
                pre_data['sys_oh'].to(DEVICE)
            ])
            in_data = gnss_data

            predict = net(in_data)
            weight = predict[0]
            bias = predict[1]

            # calculate loss
            sol_new = tas.wls_pnt_pos(sol,None,use_cache=True,w = weight,b = bias, enable_torch=True, device=DEVICE)
            gt = sol['gt']
            pos = sol_new['pos']
            enu = torch.hstack(p3d.ecef2enu(*pos[:3], gt[1], gt[2], gt[3]))+torch.tensor(gt[7:10]).to(DEVICE)
            pos_loss = torch.norm(enu[:loss_locate])

            if pos_loss > 200:
                total_len-=1
                continue

            # loss/batch_size
            loss += pos_loss
            
            accumulated_loss += pos_loss.item()
            step += 1

            # update
            if step % batch_size == 0:
                loss.backward()
                opt.step()
                opt.zero_grad()
                total_loss += accumulated_loss
                accumulated_loss = 0
                t.set_postfix({'epoch current loss': pos_loss.item()})
                loss = 0

            # 释放显存
            del gnss_data, in_data, predict, pos, enu

        if accumulated_loss > 0:
            loss.backward()
            opt.step()
            opt.zero_grad()
            total_loss += accumulated_loss

        # 计算平均损失
        avg_loss = total_loss / len(pres) if len(pres) > 0 else 0
        print(f"Epoch {k+1} Average Loss: {avg_loss}")
        vis_loss.append(avg_loss)

        # 保存checkpoint
        if (k + 1) % checkpoint_interval == 0:
            checkpoint_path = f"{checkpoint_dir}/checkpoint_epoch_{k + 1}.pth"
            torch.save(net.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")

# 最终保存模型和损失
final_model_path = f"{checkpoint_dir}/multinet_3d.pth"
torch.save(net.state_dict(), final_model_path)
print(f"Final model saved at {final_model_path}")

# 保存损失曲线
vis_loss = np.array(vis_loss)
plt.plot(vis_loss)
plt.savefig(result_path + "/loss.png")
np.savetxt(result_path + "/loss.csv", vis_loss.reshape(-1, 1))