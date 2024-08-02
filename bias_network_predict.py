import torch
import pyrtklib as prl
import rtk_util as util
import json
import sys
import numpy as np
import pandas as pd
import pymap3d as p3d
from model import BiasNet
from torch.nn import HuberLoss,MSELoss
import matplotlib.pyplot as plt


DEVICE = 'cuda'


try:
    config = sys.argv[1]
except:
    config = "config/klt2_predict.json"

with open(config) as f:
    conf = json.load(f)

mode = conf['mode']
if mode not in ['train','predict']:
    raise RuntimeError("%s is not a valid option"%mode)

net = BiasNet(torch.tensor([0,0,0],dtype=torch.float32),torch.tensor([1,1,1],dtype=torch.float32),torch.tensor(0,dtype=torch.float32),torch.tensor(1,dtype=torch.float32))
net.load_state_dict(torch.load(conf['model']+"/biasnet.pth"))
net = net.to(DEVICE)
net.eval()
obs,nav,sta = util.read_obs(conf['obs'],conf['eph'])

obss = util.split_obs(obs)

tmp = []

if conf.get("gt",None):
    gt = pd.read_csv(conf['gt'],skiprows = 30, header = None,sep =' +', skipfooter = 4, error_bad_lines=False, engine='python')
    gt[0] = gt[0]+18 # leap seconds
    gts = []

for o in obss:
    t = o.data[0].time
    t = t.time+t.sec
    if t > conf['start_time'] and (conf['end_time'] == -1 and 1 or t < conf['end_time']):
        tmp.append(o)
        if conf.get("gt",None):
            gt_row = gt.loc[(gt[0]-t).abs().argmin()]
            gts.append([gt_row[3]+gt_row[4]/60+gt_row[5]/3600,gt_row[6]+gt_row[7]/60+gt_row[8]/3600,gt_row[9]])

obss = tmp

pos_errs = []

for i in range(len(obss)):
    o = obss[i]
    if conf.get("gt",None):
        gt_row = gts[i]
    ret = util.get_ls_pnt_pos(o,nav)
    if not ret['status']:
        continue
    pos_err_src = p3d.ecef2enu(*ret['pos'][:3],gt_row[0],gt_row[1],gt_row[2])
    rs = ret['data']['eph']
    dts = ret['data']['dts']
    sats = ret['data']['sats']
    exclude = ret['data']['exclude']
    prs = ret['data']['prs']
    resd = np.array(ret['data']['residual'])
    SNR = np.array(ret['data']['SNR'])
    azel = np.delete(np.array(ret['data']['azel']).reshape((-1,2)),exclude,axis=0)
    
    in_data = torch.tensor(np.hstack([SNR.reshape(-1,1),azel[:,1:],resd]),dtype=torch.float32).to(DEVICE)


    predict_bias = net(in_data).detach().cpu().numpy()
    select_sats = list(np.delete(np.array(sats),exclude))

    for k in range(o.n):
        try:
            index = select_sats.index(o.data[k].sat)
            o.data[k].P[0] = o.data[k].P[0] - predict_bias[index]
        except ValueError:
            continue
    
    ret = util.get_ls_pnt_pos(o,nav)
    
    pos_err_pre = p3d.ecef2enu(*ret['pos'][:3],gt_row[0],gt_row[1],gt_row[2])
    pos_errs.append([np.linalg.norm(pos_err_src[:2]),np.linalg.norm(pos_err_pre[:2])])

pos_errs = np.array(pos_errs)
print(f"SPP mean error: {pos_errs[:,0].mean():.2f}, TDL mean error: {pos_errs[:,1].mean():.2f}")
plt.plot(pos_errs[:,0],label = "SPP",color="red")
plt.plot(pos_errs[:,1],label = "TDL",color="blue")
plt.legend()
plt.show()

