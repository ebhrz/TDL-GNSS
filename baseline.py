import torch
import pyrtklib as prl
import rtk_util as util
import json
import sys
import numpy as np
import pandas as pd
import pymap3d as p3d
from model import WeightNet, BiasNetTest
from torch.nn import HuberLoss,MSELoss
import matplotlib.pyplot as plt
from tqdm import tqdm
import os


DEVICE = 'cuda'


try:
    config = sys.argv[1]
except:
    config = "config/weight/klt1_predict.json"

with open(config) as f:
    conf = json.load(f)

mode = conf['mode']
if mode not in ['train','predict']:
    raise RuntimeError("%s is not a valid option"%mode)

result = config.split("/")[-1].split(".json")[0]
result_path = "result/"+result
os.makedirs(result_path,exist_ok=True)

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

pos_errs = []


errors2d = []
errors3d = []
TDL_weight_pos = []
TDL_bias_pos = []
go_pos = []
rtklib_pos = []
gt_pos = []
TDL_bw_pos = []
with tqdm(range(len(obss))) as t:
    for i in t:
        o = obss[i]
        if conf.get("gt",None):
            gt_row = gts[i]
        try:
            ret = util.get_ls_pnt_pos(o,nav)
            if not ret['status']:
                continue
        except:
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
        goGPS_weight = util.goGPSW(in_data.cpu().numpy())
        #print(predict_weight)
        select_sats = list(np.delete(np.array(sats),exclude))

        ls_pos = ret['pos']

        ret = util.get_ls_pnt_pos_torch(o,nav,w = torch.diag(torch.tensor(goGPS_weight).to(DEVICE)),p_init=ls_pos)
        go_pos.append(p3d.ecef2geodetic(*ret['pos'][:3].detach().cpu().numpy()))

        ret = util.get_wls_pnt_pos(o,nav)
        #sol,status = util.get_rtklib_pnt(o,nav,prcopt,"SPP")
        rtklib_pos.append(p3d.ecef2geodetic(*ret['pos'][:3]))

        gt_ecef = p3d.geodetic2ecef(*gt_row)
        
        errors3d.append([
            np.linalg.norm(p3d.geodetic2enu(*go_pos[-1],gt_row[0],gt_row[1],gt_row[2])),
            np.linalg.norm(p3d.geodetic2enu(*rtklib_pos[-1],gt_row[0],gt_row[1],gt_row[2]))
        ])
        
        errors2d.append([
            np.linalg.norm(p3d.geodetic2enu(*go_pos[-1],gt_row[0],gt_row[1],gt_row[2])[:2]),
            np.linalg.norm(p3d.geodetic2enu(*rtklib_pos[-1],gt_row[0],gt_row[1],gt_row[2])[:2])
        ])

        # enu = p3d.ecef2enu(*ret['pos'][:3],gt_row[0],gt_row[1],gt_row[2])
        # error2d = torch.norm(torch.hstack(enu[:2])).item()
        gt_pos.append([gt_row[0],gt_row[1],gt_row[2]])

errors2d = np.array(errors2d)
errors3d =np.array(errors3d)


# np.savetxt(result_path+"/result3d_error.csv",errors3d,delimiter=',',header="TDL-GNSS bw,TDL-GNSS weight,TDL-GNSS bias,goGPS,RTKLIB")
# np.savetxt(result_path+"/result2d_error.csv",errors2d,delimiter=',',header="TDL-GNSS bw,TDL-GNSS weight,TDL-GNSS bias,goGPS,RTKLIB")
gt_pos = np.array(gt_pos)
rtklib_pos = np.array(rtklib_pos)
go_pos = np.array(go_pos)

np.savetxt(result_path+"/gt.csv",gt_pos,delimiter=',',header="lat,lon,height",comments='')
np.savetxt(result_path+"/rtklib_pos.csv",rtklib_pos,delimiter=',',header="lat,lon,height",comments='')
np.savetxt(result_path+"/gogps_pos.csv",go_pos,delimiter=',',header="lat,lon,height",comments='')

print(f"goGPS SPP 2D error mean: {errors2d[:,0].mean():.2f}, goGPS SPP 3D error mean: {errors3d[:,0].mean():.2f}")
print(f"rtklib SPP 2D error mean: {errors2d[:,1].mean():.2f}, rtklib SPP 3D error mean: {errors3d[:,1].mean():.2f}")
