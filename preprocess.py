import json
import sys
import numpy as np
from tqdm import tqdm
import os
import pickle
from datetime import datetime
import tasgnss as tas
import math


def align_data(obss, nav, gts):
    aligned_data = []
    with tqdm(range(len(obss))) as t:
        for i in t:
            obs = obss[i]
            gt = gts[i]
            ret = tas.wls_pnt_pos(obs,nav,return_residual=True,w=1)
            if not ret['status']:
                print("no solution for this epoch")
                continue
            aligned = {'gnss': ret,'gt':leverarm_correction(gt)}
            aligned_data.append(aligned)
    
    return aligned_data


def r(d):
    return d * math.pi / 180

def leverarm_correction(gt):
    roll,pitch,heading = gt[4:7]
    yaw = -heading
    c_phi = np.cos(r(roll))
    s_phi = np.sin(r(roll))
    c_theta = np.cos(r(pitch))
    s_theta = np.sin(r(pitch))
    c_psi = np.cos(r(yaw))
    s_psi = np.sin(r(yaw))
    T_body2enu = np.array([[c_psi * c_phi - s_psi * s_theta * s_phi, -s_psi * c_theta, c_psi * s_phi + s_psi * s_theta * c_phi, 0],
                            [s_psi * c_phi + c_psi * s_theta * s_phi, c_psi * c_theta, s_psi * s_phi - c_psi * s_theta * c_phi, 0],
                            [-c_theta * s_phi, s_theta, c_theta * c_phi, 0],
                            [0, 0, 0, 1]])
    arm_local_tmp = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0.86],
                              [0, 0, 1, -0.31],
                              [0, 0, 0, 1]])
    arm_local = np.dot(T_body2enu, arm_local_tmp)
    gt = np.hstack([gt,arm_local[:3,3]])
    return gt



ctime = datetime.now().strftime("%Y-%m-%d_%H_%M")


try:
    config = sys.argv[1]
except:
    config = "config/whampoa_0714.json"

with open(config) as f:
    conf = json.load(f)

mode = conf['mode']
if mode not in ['train','predict']:
    raise RuntimeError("%s is not a valid option"%mode)

result_path = conf['save_path']

os.makedirs(result_path,exist_ok=True)

prefix = conf['prefix']+"/"


sols = []


if "KLT" in prefix:
    for kltconfig in conf['datasets']:
        with open(kltconfig) as csf:
            kltconf = json.load(csf)
            gt = np.loadtxt(prefix+kltconf['gt'].replace("gt.csv","gts_rot.csv"), delimiter=',')
            obs,nav,sta = tas.read_obs(prefix+kltconf['files'][0],prefix+kltconf['files'][1])
            obss = tas.split_obs(obs)
            obss = tas.filter_obs(obss, kltconf['start_utc'], kltconf['end_utc'])
            if len(obss) != len(gt):
                exit("obs and gt length mismatch")
            aligned_data = align_data(obss, nav, gt)
            sols.extend(aligned_data)
            print("done with %s"%kltconfig)
else:
    gt = np.loadtxt(prefix+conf['gt'], delimiter=',')
    obs,nav,sta = tas.read_obs(prefix+conf['obs'][0],[prefix+c for c in conf['obs'][1:]])
    obss = tas.split_obs(obs)
    obss = tas.filter_obs(obss, conf['start_utc'], conf['end_utc'])
    assert len(obss) == len(gt), "obs and gt length mismatch"
    aligned_data = align_data(obss, nav, gt)
    sols.extend(aligned_data)
    print("done")

with open(result_path+"/preprocess.pkl","wb") as f:
    pickle.dump(sols,f)