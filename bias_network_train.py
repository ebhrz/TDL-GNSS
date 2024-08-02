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
from tqdm import tqdm

DEVICE = 'cuda'


try:
    config = sys.argv[1]
except:
    config = "config/klt3_train.json"

with open(config) as f:
    conf = json.load(f)

mode = conf['mode']
if mode not in ['train','predict']:
    raise RuntimeError("%s is not a valid option"%mode)

obs,nav,sta = util.read_obs(conf['obs'],conf['eph'])

obss = util.split_obs(obs)

tmp = []

if conf.get("gt",None):
    gt = pd.read_csv(conf['gt'],skiprows = 30, header = None,sep =' +', skipfooter = 4, error_bad_lines=False, engine='python')
    gt[0] = gt[0]+18 # leap seconds
    gts = []
else:
    raise RuntimeError("Please provide ground truth")

for o in obss:
    t = o.data[0].time
    t = t.time+t.sec
    if t > conf['start_time'] and (conf['end_time'] == -1 and 1 or t < conf['end_time']):
        tmp.append(o)
        if conf.get("gt",None):
            gt_row = gt.loc[(gt[0]-t).abs().argmin()]
            gts.append([gt_row[3]+gt_row[4]/60+gt_row[5]/3600,gt_row[6]+gt_row[7]/60+gt_row[8]/3600,gt_row[9]])

obss = tmp

in_data = []

out_data = []


for i in range(len(obss)):
    o = obss[i]
    if conf.get("gt",None):
        gt_row = gts[i]
    ret = util.get_ls_pnt_pos(o,nav)
    if not ret['status']:
        continue
    rs = ret['data']['eph']
    dts = ret['data']['dts']
    sats = ret['data']['sats']
    exclude = ret['data']['exclude']
    prs = ret['data']['prs']
    resd = ret['data']['residual']
    SNR = ret['data']['SNR']
    gt_ecef = p3d.geodetic2ecef(*gt_row)
    gt_p = np.hstack([gt_ecef,ret['pos'][3:]])
    H,R,azel,ex,sysinfo,count,vels,vions,vtrps = util.H_matrix_prl(rs,gt_p,dts,o.data[0].time,nav,sats,exclude)
    label = prs-R
    skip = 0
    azel = np.delete(np.array(azel).reshape((-1,2)),ex,axis=0)
    out_data.append(label)
    in_data.append(np.hstack([SNR.reshape(-1,1),azel[:,1:],resd]))
    
in_data = np.vstack(in_data)
out_data = np.vstack(out_data)
    
train = torch.tensor(in_data,dtype=torch.float32).to(DEVICE)
label = torch.tensor(out_data,dtype=torch.float32).to(DEVICE)



net = BiasNet(train.mean(axis=0),train.std(axis=0),label.mean(),label.std()).to(DEVICE)

opt = torch.optim.Adam(net.parameters(),lr = 0.01)
epoch = conf.get('epoch',500)
batch = conf.get('batch',128)
t_num = train.shape[0]
t_batch = t_num//batch+1
s_index = list(range(t_num))
lossFn = MSELoss(reduction='sum')
delta = 0
vis_loss = []

for k in range(epoch):
    ep_loss = 0
    with tqdm(range(t_batch),desc=f"Epoch {k+1}") as t:
        for i in t:
            start = i*batch
            end = start+batch
            batch_input = train[start:end]
            pre = net(batch_input)
            batch_label = label[start:end]
            loss = lossFn(pre,batch_label)
            opt.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            opt.step()
            ep_loss+=loss.item()
            t.set_postfix({'batch mean loss':loss.item()/t_batch})
            #t.set_postfix({'batch mean loss':loss.item()/t_batch*o_std+o_mean})
    m_loss = ep_loss/t_num
    print('epoch:%d,mean loss:%.2f'%(k+1,m_loss))
    vis_loss.append(m_loss)
torch.save(net.state_dict(),conf['model']+"/biasnet.pth")