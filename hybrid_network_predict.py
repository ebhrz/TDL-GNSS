import torch
import pyrtklib as prl
import rtk_util as util
import json
import sys
import numpy as np
import pandas as pd
import pymap3d as p3d
from model import WeightNet, BiasNetTest,HybridShareNet
from torch.nn import HuberLoss,MSELoss
import matplotlib.pyplot as plt
from tqdm import tqdm
import os


DEVICE = 'cuda'


try:
    config = sys.argv[1]
except:
    config = "config/bias/whampoa_predict.json"

with open(config) as f:
    conf = json.load(f)

mode = conf['mode']
if mode not in ['train','predict']:
    raise RuntimeError("%s is not a valid option"%mode)

result = config.split("/")[-1].split(".json")[0]
result_path = "result/"+result
os.makedirs(result_path,exist_ok=True)
os.makedirs(result_path+"/bw",exist_ok=True)
net = HybridShareNet()
net.double()
#net.load_state_dict(torch.load("model/hybrid_share/hybrid_share.pth"))
#net.load_state_dict(torch.load("model/hybrid_share_new/hybrid_share_100.pth"))
net.load_state_dict(torch.load("model/hybrid_share_p40/hybrid_share_150.pth"))
net = net.to(DEVICE)


obs,nav,sta = util.read_obs(conf['obs'],conf['eph'])
prl.sortobs(obs)
prcopt = prl.prcopt_default
obss = util.split_obs(obs)

tmp = []

if conf.get("gt",None):
    gt = pd.read_csv(conf['gt'],skiprows = 30, header = None,sep =' +', skipfooter = 4, error_bad_lines=False, engine='python')
    #gt = pd.read_csv(conf['gt'],skiprows = 2, header = None,sep =' +', skipfooter = 4, error_bad_lines=False, engine='python')
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
net.eval()
errors = []
gt_pos = []
TDL_bw_pos = []
ecef_pos = []
samples = 0
with tqdm(range(len(obss))) as t:
    for i in t:
        o = obss[i]
        if conf.get("gt",None):
            gt_row = gts[i]
        try:
            ret = util.get_ls_pnt_pos(o,nav)
            if not ret['status']:
                print(ret['msg'])
                continue
        except Exception as e:
            print(e)
            continue

        rs = ret['data']['eph']
        dts = ret['data']['dts']
        sats = ret['data']['sats']
        exclude = ret['data']['exclude']
        prs = ret['data']['prs']
        resd = np.array(ret['data']['residual'])
        SNR = np.array(ret['data']['SNR'])
        azel = np.delete(np.array(ret['data']['azel']).reshape((-1,2)),exclude,axis=0)
        in_data = torch.tensor(np.hstack([SNR.reshape(-1,1),azel[:,1:],resd]),dtype=torch.float32).to(DEVICE)
        
        samples+=in_data.shape[0]

        predict= net(in_data)
        weight = predict[0]
        bias = predict[1]
        
        sats_used = np.delete(np.array(sats),exclude,axis=0)
        snp = sats_used
        wnp = weight.detach().cpu().numpy()
        bnp = bias.detach().cpu().numpy()
        ep = pd.DataFrame(np.vstack([snp,wnp,bnp]).T)
        ep.columns=['sat','weight','bias']
        ep.to_csv(result_path+"/bw/%d.csv"%i,index=None)

        ret = util.get_ls_pnt_pos_torch(o,nav,torch.diag(weight),bias.reshape(-1,1),p_init=ret['pos'])
        TDL_bw_pos.append(p3d.ecef2geodetic(*ret['pos'][:3].detach().cpu().numpy()))
        gt_pos.append([gt_row[0],gt_row[1],gt_row[2]])
        errors.append(p3d.geodetic2enu(*TDL_bw_pos[-1],*gt_pos[-1]))
        ecef_pos.append(ret['pos'].detach().cpu().numpy())

ecef_pos = np.array(ecef_pos)
gt_pos = np.array(gt_pos)
TDL_bw_pos = np.array(TDL_bw_pos)
errors = np.array(errors)
np.savetxt(result_path+"/ecef.csv",ecef_pos,delimiter=',',header="x,y,z,t1,t2,t3,t4",comments='')
np.savetxt(result_path+"/gt.csv",gt_pos,delimiter=',',header="lat,lon,height",comments='')
np.savetxt(result_path+"/TDL_bw_pos.csv",TDL_bw_pos,delimiter=',',header="lat,lon,height",comments='')
print(f"2D mean: {np.linalg.norm(errors[:,:2],axis=1).mean():.2f}, 3D mean: {np.linalg.norm(errors,axis=1).mean():.2f}")
print(f"Samples {samples}")