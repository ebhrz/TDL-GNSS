import pyrtklib as prl
#import pyrtklib_debug.build.pyrtklib_debug as prl
#import pyrtklib_debug.build.pyrtklib as prl
import numpy as np
import math
import os
try:
    import torch
    torch_enable = True
except:
    torch_enable = False    



SYS = {'G':prl.SYS_GPS,'C':prl.SYS_CMP,'E':prl.SYS_GAL,'R':prl.SYS_GLO,'J':prl.SYS_QZS}

def memorize_func(func):
    cache = {}
    def wrapper(index,*args):
        if index in cache:
            return cache[index]
        else:
            result = func(*args)
            cache[index] = result
            return result
    return wrapper

def arr_select(arr,select,step = 1):
    obj_class = type(arr)
    n = len(select)*step
    arr_sel = obj_class(n)
    for i in range(len(select)):
        for j in range(step):
            arr_sel[i*step+j] = arr[select[i]*step+j]
    return arr_sel

def arr(src_list,arr_type):
    l = len(src_list)
    ret = arr_type(l)
    for i in range(l):
        ret[i]=src_list[i]
    return ret

def gettgd(sat, nav, type):
    sys_name = prl.Arr1Dchar(4)
    prl.satno2id(sat,sys_name)
    sys = SYS[sys_name.ptr[0]]
    eph = nav.eph
    geph = nav.geph
    if sys == prl.SYS_GLO:
        for i in range(nav.ng):
            if geph[i].sat == sat:
                break
        return 0.0 if i >= nav.ng else -geph[i].dtaun * prl.CLIGHT
    else:
        for i in range(nav.n):
            if eph[i].sat == sat:
                break
        return 0.0 if i >= nav.n else eph[i].tgd[type] * prl.CLIGHT

def prange(obs, nav, opt, var):
    P1, P2, gamma, b1, b2 = 0.0, 0.0, 0.0, 0.0, 0.0
    var[0] = 0.0

    sat = obs.sat

    sys_name = prl.Arr1Dchar(4)
    prl.satno2id(sat,sys_name)
    sys = SYS[sys_name.ptr[0]]
    P1 = obs.P[0]
    P2 = obs.P[1]

    if P1 == 0.0 or (opt.ionoopt == prl.IONOOPT_IFLC and P2 == 0.0):
        return 0.0

    # P1-C1, P2-C2 DCB correction
    if sys == prl.SYS_GPS or sys == prl.SYS_GLO:
        if obs.code[0] == prl.CODE_L1C:
            P1 += nav.cbias[sat - 1,1]  # C1->P1
        if obs.code[1] == prl.CODE_L2C:
            P2 += nav.cbias[sat - 1,2]  # C2->P2

    if opt.ionoopt == prl.IONOOPT_IFLC:  # dual-frequency
        if sys == prl.SYS_GPS or sys == prl.SYS_QZS:  # L1-L2, G1-G2
            gamma = (prl.FREQ1 / prl.FREQ2) ** 2
            return (P2 - gamma * P1) / (1.0 - gamma)
        elif sys == prl.SYS_GLO:  # G1-G2
            gamma = (prl.FREQ1_GLO / prl.FREQ2_GLO) ** 2
            return (P2 - gamma * P1) / (1.0 - gamma)
        elif sys == prl.SYS_GAL:  # E1-E5b
            gamma = (prl.FREQ1 / prl.FREQ7) ** 2
            if prl.getseleph(prl.SYS_GAL):  # F/NAV
                P2 -= gettgd(sat, nav, 0) - gettgd(sat, nav, 1)  # BGD_E5aE5b
            return (P2 - gamma * P1) / (1.0 - gamma)
        elif sys == prl.SYS_CMP:  # B1-B2
            gamma = (((prl.FREQ1_CMP if obs.code[0] == prl.CODE_L2I else prl.FREQ1) / prl.FREQ2_CMP) ** 2)
            b1 = gettgd(sat, nav, 0) if obs.code[0] == prl.CODE_L2I else gettgd(sat, nav, 2) if obs.code[0] == prl.CODE_L1P else gettgd(sat, nav, 2) + gettgd(sat, nav, 4)  # TGD_B1I / TGD_B1Cp / TGD_B1Cp+ISC_B1Cd
            b2 = gettgd(sat, nav, 1)  # TGD_B2I/B2bI (m)
            return ((P2 - gamma * P1) - (b2 - gamma * b1)) / (1.0 - gamma)
        elif sys == prl.SYS_IRN:  # L5-S
            gamma = (prl.FREQ5 / prl.FREQ9) ** 2
            return (P2 - gamma * P1) / (1.0 - gamma)
    else:  # single-freq (L1/E1/B1)
        var[0] = 0.3 ** 2
        
        if sys == prl.SYS_GPS or sys == prl.SYS_QZS:  # L1
            b1 = gettgd(sat, nav, 0)  # TGD (m)
            return P1 - b1
        elif sys == prl.SYS_GLO:  # G1
            gamma = (prl.FREQ1_GLO / prl.FREQ2_GLO) ** 2
            b1 = gettgd(sat, nav, 0)  # -dtaun (m)
            return P1 - b1 / (gamma - 1.0)
        elif sys == prl.SYS_GAL:  # E1
            b1 = gettgd(sat, nav, 0) if prl.getseleph(prl.SYS_GAL) else gettgd(sat, nav, 1)  # BGD_E1E5a / BGD_E1E5b
            return P1 - b1
        elif sys == prl.SYS_CMP:  # B1I/B1Cp/B1Cd
            b1 = gettgd(sat, nav, 0) if obs.code[0] == prl.CODE_L2I else gettgd(sat, nav, 2) if obs.code[0] == prl.CODE_L1P else gettgd(sat, nav, 2) + gettgd(sat, nav, 4)  # TGD_B1I / TGD_B1Cp / TGD_B1Cp+ISC_B1Cd
            return P1 - b1
        elif sys == prl.SYS_IRN:  # L5
            gamma = (prl.FREQ9 / prl.FREQ5) ** 2
            b1 = gettgd(sat, nav, 0)  # TGD (m)
            return P1 - gamma * b1
    return P1

def nextobsf(obs,i):
    n = 0
    while i+n < obs.n:
        tt = prl.timediff(obs.data[i+n].time,obs.data[i].time)
        if tt > 0.05:
            break
        n+=1
    return n

def get_sat_cnt(obs,i,m,rcv):
    cnt = 0
    for k in range(i,i+m):
        if obs.data[k].rcv == rcv:
            cnt+=1
    return cnt

# deprecated
# def get_sat_pos(obsd,n,nav):
#     svh = prl.Arr1Dint(prl.MAXOBS)
#     rs = prl.Arr1Ddouble(6*n)
#     dts = prl.Arr1Ddouble(2*n)
#     var = prl.Arr1Ddouble(1*n)
#     prl.satposs(obsd[0].time,obsd.ptr,n,nav,0,rs,dts,var,svh)
#     noeph = []
#     for i in range(n):
#         if rs[6*i] == 0:
#             noeph.append(i)
#     nrs = prl.Arr1Ddouble(6*(n-len(noeph)))
#     j = 0
#     for i in range(6*n):
#         if rs[i]!=0:
#             nrs[j] = rs[i]
#             j+=1
#     return nrs,noeph

def split_obs(obs):
    i = 0
    m = nextobsf(obs,i)
    obss = []
    while m!=0:
        tmp_obs = prl.obs_t()
        tmp_obs.data = prl.Arr1Dobsd_t(m)
        rcv1 = 0
        rcv2 = 0
        for j in range(m):
            if obs.data[i+j].rcv == 1:
                tmp_obs.data[rcv1] = obs.data[i+j]
                rcv1+=1
        if rcv1 == 0:
            i+=m
            m = nextobsf(obs,i)
            continue
        if rcv1 != m:
            for j in range(m-rcv1):
                if obs.data[i+j+rcv1].rcv == 2:
                    tmp_obs.data[rcv1+rcv2] = obs.data[i+rcv1+j]
                    rcv2+=1
        tmp_obs.n = m
        tmp_obs.nmax = m
        i+=m
        obss.append(tmp_obs)
        m = nextobsf(obs,i)
    return obss


def get_common_obs(obs):
    m = obs.n
    tmp1 = {}
    tmp2 = {}
    for i in range(m):
        if obs.data[i].rcv == 1:
            tmp1[obs.data[i].sat-1] = obs.data[i]
        else:
            tmp2[obs.data[i].sat-1] = obs.data[i]
    common = []
    index = []
    tmp1keys = list(tmp1.keys())
    for j in range(len(tmp1keys)):
        i = tmp1keys[j]
        if i in tmp2:
            if tmp1[i].P[0]!=0 and tmp2[i].P[0]!=0:
                common.append(i)
                index.append(j)
    obsd1 = prl.Arr1Dobsd_t(len(common))
    obsd2 = prl.Arr1Dobsd_t(len(common))
    for i in range(len(common)):
        obsd1[i] = tmp1[common[i]]
        obsd2[i] = tmp2[common[i]]
    return obsd1,obsd2,common,index


def rov2head(obs,n):
    r_head = prl.Arr1Dobsd_t(n)
    j = 0
    for i in range(n):
        if obs[i].rcv == 1:
            r_head[j] = obs[i]
            j+=1
    for i in range(n):
        if obs[i].rcv == 2:
            r_head[j] = obs[i]
            j+=1
    return r_head

def get_sat_id(no):
    tmp = prl.Arr1Dchar(4)
    prl.satno2id(no,tmp)
    return tmp[0]

def get_max_sys_el(obsd,n,sol_spp):
    azel = sol_spp['data']['azel']
    ex = sol_spp['data']['exclude']
    mask = list(set(range(n))-set(ex))
    azel = arr_select(azel,mask,2)
    ms = {sys:[-1,0] for sys in SYS}
    for i in range(n):
        name = get_sat_id(obsd[i].sat)
        sys = name[0]
        el = azel[i*2+1]
        if el > ms[sys][1]:
            ms[sys][0] = i
            ms[sys][1] = el
    return ms



def get_rtklib_pnt(obs,nav,prcopt,mode):
    if mode == "SPP":
        sol,sat,azel,msg = get_obs_pnt(obs,nav,prcopt)
        if msg.ptr and "chi-square error" not in msg.ptr and "gdop error" not in msg.ptr:
            return sol,False
        else:
            return sol,True
    if mode == "DGNSS":
        rtk = get_obs_rtk(obs,nav,prcopt)
        # return rtk.sol,True
        msg = rtk.errbuf.ptr
        if msg and "ambiguity validation failed" not in msg and "chi-square error" not in msg and "slip detected forward" not in msg and "slip detected half-cyc" not in msg and "no double-differenced residual" not in msg:
            return rtk.sol,False
        else:
            return rtk.sol,True

# this calls the pntpos in rtklib
def get_obs_pnt(obs,nav,prcopt):
    m = obs.n
    sol = prl.sol_t()
    sat = prl.Arr1Dssat_t(prl.MAXSAT)
    sol.time = obs.data[0].time
    msg = prl.Arr1Dchar(100)
    azel = prl.Arr1Ddouble(m*2)
    prl.pntpos(obs.data.ptr,obs.n,nav,prcopt,sol,azel,sat.ptr,msg)
    return sol,sat,azel,msg

def get_obs_rtk(obs,nav,prcopt):
    rtk = prl.rtk_t()
    prl.rtkinit(rtk,prcopt)
    #rtk_obs = rov2head(obs.data,obs.n)
    prl.rtkpos(rtk,obs.data.ptr,obs.n,nav)
    return rtk


def read_obs(rcv,eph,ref=None):
    obs = prl.obs_t()
    nav = prl.nav_t()
    sta = prl.sta_t()
    if type(rcv) is list:
        for r in rcv:
            prl.readrnx(r,1,"",obs,nav,sta)
    else:
        prl.readrnx(rcv,1,"",obs,nav,sta)
    if type(eph) is list:
        for f in eph:
            prl.readrnx(f,1,"",obs,nav,sta)
    else:
        prl.readrnx(eph,1,"",obs,nav,sta)
    if ref:
        if type(ref) is list:
            for r in ref:
                prl.readrnx(r,2,"",obs,nav,sta)
        else:
            prl.readrnx(ref,2,"",obs,nav,sta)
    return obs,nav,sta
        
def get_obs_utc_time(obstime):
    return obstime.time+obstime.sec-18

def get_obs_time(obstime):
    return obstime.time+obstime.sec

def H_matrix_prl(satpos,rp,dts,time,nav,sats,exclude = []):
    ex = [] # exclude sat order
    n = int(len(satpos)/6)
    Ht = []
    Rt = []
    e = prl.Arr1Ddouble(3)
    rr = prl.Arr1Ddouble(3)
    rr[0] = rp[0]
    rr[1] = rp[1]
    rr[2] = rp[2]
    azel = prl.Arr1Ddouble(n*2)
    pos = prl.Arr1Ddouble(3)
    prl.ecef2pos(rr,pos)
    
    dion = prl.Arr1Ddouble(1)
    vion = prl.Arr1Ddouble(1)
    dtrp = prl.Arr1Ddouble(1)
    vtrp = prl.Arr1Ddouble(1)

    vels = []
    vions = []
    vtrps = []

    sysname = prl.Arr1Dchar(4)
    count = {'G':0,'C':0,'E':0,'R':0}
    for i in range(n):
        if i in exclude:
            ex.append(i)
            continue
        sp = np.array(satpos[i*6+0:i*6+3])
        #r = np.linalg.norm(sp-rp[:3])
        r = prl.geodist(satpos[i*6:i*6+3],rr,e)
        azel_tmp = prl.Arr1Ddouble(2)
        prl.satazel(pos,e,azel_tmp)
        azel[i*2] = azel_tmp[0]
        azel[i*2+1] = azel_tmp[1]

        if azel_tmp[1] < 0:
            ex.append(i)
            continue
        prl.ionocorr(time,nav,sats[i],pos,azel_tmp,prl.IONOOPT_BRDC,dion,vion)
        prl.tropcorr(time,nav,pos,azel_tmp,prl.TROPOPT_SAAS,dtrp,vtrp)
        vions.append(vion.ptr)
        vtrps.append(vtrp.ptr)
        prl.satno2id(sats[i],sysname)
        vel = varerr(azel_tmp[1],sysname.ptr[0])
        vels.append(vel)
        if 'G' in sysname.ptr:
            Ht.append([-(sp[0]-rp[0])/r,-(sp[1]-rp[1])/r,-(sp[2]-rp[2])/r,1,0,0,0])
            Rt.append(r+rp[3]-prl.CLIGHT*dts[i*2]+dtrp.ptr+dion.ptr)
            count['G']+=1
        elif 'C' in sysname.ptr:
            Ht.append([-(sp[0]-rp[0])/r,-(sp[1]-rp[1])/r,-(sp[2]-rp[2])/r,0,1,0,0])
            Rt.append(r+rp[4]-prl.CLIGHT*dts[i*2]+dtrp.ptr+dion.ptr)
            count['C']+=1
        elif 'E' in sysname.ptr:
            Ht.append([-(sp[0]-rp[0])/r,-(sp[1]-rp[1])/r,-(sp[2]-rp[2])/r,0,0,1,0])
            Rt.append(r+rp[5]-prl.CLIGHT*dts[i*2]+dtrp.ptr+dion.ptr)
            count['E']+=1
        elif 'R' in sysname.ptr:
            Ht.append([-(sp[0]-rp[0])/r,-(sp[1]-rp[1])/r,-(sp[2]-rp[2])/r,0,0,0,1])
            Rt.append(r+rp[6]-prl.CLIGHT*dts[i*2]+dtrp.ptr+dion.ptr)
            count['R']+=1
    if n-len(ex) > 3:
        H = np.vstack(Ht).astype(np.float64)
        R = np.vstack(Rt).astype(np.float64)
    else:
        H = np.zeros((1,7))
        R = np.zeros((1,1))
        #return H,R,azel,ex,np.array([]),count,vels,vions,vtrps
    sysinfo = np.where(np.any(H!=0,axis=0))[0]
    H = H[:,sysinfo]
    return H,R,azel,ex,sysinfo,count,vels,vions,vtrps

def wls_solve(H,z,w = None,b = None):
    if w is None:
        w = np.eye(len(H))
    if b is None:
        b = np.zeros((len(H),1))
    t1 = np.matmul(H.T,w)
    t2 = np.matmul(t1,H)
    t3 = np.matmul(np.linalg.inv(t2),H.T)
    t4 = np.matmul(t3,w)
    x = np.matmul(t4,(z-b))
    return x


def get_dgnss_pos(obs,sta,nav):
    maxiter = 20
    ms_info = {i:[] for i in SYS}
    MS = {i:-1 for i in SYS}
    n_o1 = np.sum(np.array([obs.data[i].rcv for i in range(obs.n)])==1)
    o1,o2,com,index1 = get_common_obs(obs)
    rs, noeph, dts, var = get_sat_pos(o1,len(com),nav)
    if noeph:
        mask = list(set(range(len(com)))-set(noeph))
        o1 = arr_select(o1,mask)
        o2 = arr_select(o2,mask)
        com = list(np.array(com)[mask])
    o1_obs = prl.obs_t()
    o1_obs.data = obs.data[:n_o1]
    o1_obs.n = n_o1
    o1_obs.nmax = n_o1
    spp_sol = get_wls_pnt_pos(o1_obs,nav)
    ms = get_max_sys_el(o1,len(com),spp_sol)
    res = []
    sys_map = []
    for i in range(len(com)):
        name = get_sat_id(o1[i].sat)
        sys = name[0]
        no = ms[sys][0]
        if i == no:
            sys_map.append(sys)
            continue
        r21 = np.linalg.norm(np.array(rs[no*6+0:no*6+3])-np.array(sta.pos)).astype('float64')
        r22 = np.linalg.norm(np.array(rs[i*6+0:i*6+3])-np.array(sta.pos)).astype('float64')
        dp1 = o1[i].P[0] - o1[no].P[0]
        dp2 = o2[i].P[0] - o2[no].P[0]
        r = dp1-dp2+(r22-r21)
        res.append(r)
        sys_map.append(sys)
    res = np.array(res)
    dx = np.array([1000,1000,1000],dtype=np.double)
    p = np.array([0,0,0],dtype=np.double)
    i = 0
    while np.linalg.norm(dx) > 0.01 and i < maxiter:
        H,R = DD_H_matrix(rs,p,ms,sys_map)
        resd = res.reshape((-1,1))-R
        dx = wls_solve(H,resd)
        p = p+dx.squeeze()
        i+=1
    return p,resd
    

        
def DD_H_matrix(satpos,p00,ms,sys_map):
    n = int(len(satpos)/6)
    Ht = []
    Rt = []
    b_flag = False
    msns = [ms[sys][0] for sys in ms]
    for i in range(n):
        if i in msns:
            continue
        msn = ms[sys_map[i]][0]
        p0 = np.array(satpos[msn*6+0:msn*6+3])
        r0 = np.linalg.norm(p0-p00)
        p = np.array(satpos[i*6+0:i*6+3])
        r = np.linalg.norm(p-p00)
        Ht.append(np.array([-(p[0]-p00[0])/r-(-(p0[0]-p00[0]))/r0,-(p[1]-p00[1])/r-(-(p0[1]-p00[1]))/r0,-(p[2]-p00[2])/r-(-((p0[2]-p00[2]))/r0)]))
        Rt.append(r-r0)
    H = np.vstack(Ht)
    R = np.vstack(Rt)
    return H,R



def get_wls_pnt_pos(o,nav,exsatids=[], SNR = 0, EL = 0, RESD = 10000):
    maxiter = 20
    if o.n < 4:
        return {"status":False,"pos":np.array([0,0,0,0]),"msg":"no enough observations","data":{}}
    rs,noeph,dts,var = get_sat_pos(o.data,o.n,nav)
    w = 1/np.sqrt(np.array(var))
    # if noeph :
    #     print('some sats without ephemeris')
    prs = []
    exclude = []
    sats = []
    opt = prl.prcopt_default
    skip = 0
    sysname = prl.Arr1Dchar(4)
    vmeas = prl.Arr1Ddouble(1)
    for ii in range(o.n):
        if ii in noeph:
            skip+=1
            continue
        obsd = o.data[ii]
        pii = obsd.P[0] #only process L1
        prl.satno2id(obsd.sat,sysname)
        if  pii == 0 or obsd.sat in exsatids or sysname.ptr[0] not in ['G','C','R','E'] or obsd.SNR[0] < SNR*1e3:
            exclude.append(ii-skip)
            prs.append(0)
            sats.append(obsd.sat)
            continue
        pii = prange(obsd,nav,opt,vmeas)
        prs.append(pii)
        sats.append(obsd.sat)
        var[ii-skip] += vmeas.ptr
    prs = np.vstack(prs)
    p = np.array([0,0,0,0,0,0,0],dtype=np.float64)
    dp = np.array([100,100,100],dtype=np.float64)
    iii = 0 
    var = np.array(var)
    #W = np.eye(o.n-len(exclude))
    b = np.zeros((o.n-len(exclude),1))
    while np.linalg.norm(dp)>0.0001 and iii < maxiter:
        H,R,azel,ex,sysinfo,syscount,vels,vions,vtrps = H_matrix_prl(rs,p,dts,o.data[0].time,nav,sats,exclude)
        inc = set(range(o.n-len(noeph)))-set(ex)
        if len(inc) < 4:
            iii+=1
            continue
        tmp_var = np.delete(var,ex)
        tmp_var = tmp_var+np.array(vels)+np.array(vions)+np.array(vtrps)
        w = 1/np.sqrt(tmp_var)
        W = np.diag(w)
        resd = prs[list(inc)] - R
        dp = wls_solve(H,resd,W)
        p[sysinfo] = p[sysinfo]+dp.squeeze()
        iii+=1

    if iii >= maxiter:
        return {"status":False,"pos":p,"msg":"over max iteration times","data":{}}
    
    azel = np.delete(np.array(azel).reshape((-1,2)),ex,axis=0)
    mask = np.squeeze(abs(resd)<RESD)&(azel[:,1]>(EL*prl.D2R))
    dp = np.array([100,100,100])
    iii = 0 
    while np.linalg.norm(dp)>0.0001 and iii < maxiter:
        H,R,azel,ex,sysinfo,syscount,vels,vions,vtrps = H_matrix_prl(rs,p,dts,o.data[0].time,nav,sats,exclude)
        inc = set(range(o.n-len(noeph)))-set(ex)
        if len(inc) < 4:
            iii+=1
            continue
        tmp_var = np.delete(var,ex)
        tmp_var = tmp_var+np.array(vels)+np.array(vions)+np.array(vtrps)
        w = 1/np.sqrt(tmp_var)
        resd = prs[list(inc)] - R

        H = H[mask]
        if H.shape[0] < H.shape[1]:
            return {"status":False,"pos":p,"msg":"no enough observations","data":{}}
        sysinfo_filter = np.where(np.any(H!=0,axis=0))[0]
        H = H[:,sysinfo_filter]
        resd = resd[mask]
        w = w[mask]
        W = np.diag(w)

        dp = wls_solve(H,resd,W)
        p[sysinfo[sysinfo_filter]] = p[sysinfo[sysinfo_filter]]+dp.squeeze()
        iii+=1
    
    if np.sqrt((resd*resd).sum()) > 1000:
        return {"status":False,"pos":p,"msg":"residual too large","data":{"residual":resd}}
    H,R,azel,ex,sysinfo,syscount,vels,vions,vtrps = H_matrix_prl(rs,p,dts,o.data[0].time,nav,sats,exclude)
    h = H[:,:3]
    q = np.linalg.inv(np.matmul(h.T,h))
    pdop = np.sqrt(q[0,0]+q[1,1]+q[2,2])
    return {"status":True,"pos":p,"msg":"success","data":{"residual":resd,'azel':azel,"exclude":ex,"eph":rs,'PDOP':pdop}}

def get_sat_pos(obsd,n,nav, SRC_dts = False):
    svh = prl.Arr1Dint(prl.MAXOBS)
    rs = prl.Arr1Ddouble(6*n)
    dts = prl.Arr1Ddouble(2*n)
    var = prl.Arr1Ddouble(1*n)
    prl.satposs(obsd[0].time,obsd.ptr,n,nav,0,rs,dts,var,svh)
    noeph = []
    for i in range(n):
        if abs(rs[6*i]) < 1e-10:
            noeph.append(i)
    mask = list(set(range(n))-set(noeph))
    nrs = arr_select(rs,mask,6)
    var = arr_select(var,mask)
    if not SRC_dts:
        ndts = arr_select(dts,mask,2)
    else:
        ndts = dts
    # nrs = prl.Arr1Ddouble(6*(n-len(noeph)))
    # ndts = prl.Arr1Ddouble(2*(n-len(noeph)))
    # j = 0
    # for i in range(6*n):
    #     if rs[i]!=0:
    #         nrs[j] = rs[i]
    #         j+=1
    # j = 0
    # for i in range(2*n):
    #     if rs[i]!=0:
    #         ndts[j] = dts[i]
    #         j+=1
    return nrs,noeph,ndts,var

def varerr(el,sys):
    if sys in ['R']:
        fact = 1.5
    else:
        fact = 1
    if el < 5*prl.D2R:
        el = 5*prl.D2R
    err = prl.prcopt_default.err
    varr=(err[0]**2)*((err[1]**2)+(err[2])/np.sin(el))
    return (fact**2)*varr

def get_nlos_wls_pnt_pos(o,nav,nlos,exsatids=[]):
    maxiter = 10
    if o.n < 4:
        return {"status":False,"pos":np.array([0,0,0,0]),"msg":"no enough observations","data":{}}
    rs,noeph,dts,var = get_sat_pos(o.data,o.n,nav)
    w = 1/np.sqrt(np.array(var))
    # if noeph :
    #     print('some sats without ephemeris')
    prs = []
    exclude = []
    sats = []
    opt = prl.prcopt_default
    skip = 0
    sysname = prl.Arr1Dchar(4)
    vmeas = prl.Arr1Ddouble(1)
    for ii in range(o.n):
        #print(ii,o.n)
        if ii in noeph:
            skip+=1
            continue
        obsd = o.data[ii]
        pii = obsd.P[0] #only process L1
        prl.satno2id(obsd.sat,sysname)
        if  pii == 0 or obsd.sat in exsatids or sysname.ptr[0] not in ['G','C','R','E']:
            exclude.append(ii-skip)
            prs.append(0)
            sats.append(obsd.sat)
            continue
        pii = prange(obsd,nav,opt,vmeas)
        prs.append(pii)
        sats.append(obsd.sat)
        var[ii-skip] += vmeas.ptr
    if len(prs)<4:
        return {"status":False,"pos":np.array([0,0,0,0]),"msg":"no enough observations","data":{}}
    prs = np.vstack(prs)
    p = np.array([0,0,0,0,0,0,0],dtype=np.float64)
    dp = np.array([100,100,100],dtype=np.float64)
    iii = 0 
    var = np.array(var)
    #W = np.eye(o.n-len(exclude))
    b = np.zeros((o.n-len(exclude),1))
    while np.linalg.norm(dp)>0.0001 and iii < maxiter:
        H,R,azel,ex,sysinfo,syscount,vels,vions,vtrps = H_matrix_prl_nlos(rs,p,dts,o.data[0].time,nav,sats,nlos,exclude)
        inc = set(range(o.n-len(noeph)))-set(ex)
        if len(inc) < 4:
            iii+=1
            continue
        tmp_var = np.delete(var,ex)
        tmp_var = tmp_var+np.array(vels)+np.array(vions)+np.array(vtrps)
        w = 1/np.sqrt(tmp_var)
        W = np.diag(w)
        resd = prs[list(inc)] - R
        dp = wls_solve(H,resd,W)
        p[sysinfo] = p[sysinfo]+dp.squeeze()
        iii+=1
    if iii >= 10:
        return {"status":False,"pos":p,"msg":"over max iteration times","data":{"residual":resd}}
    if np.sqrt((resd*resd).sum()) > 1000:
        return {"status":False,"pos":p,"msg":"residual too large","data":{"residual":resd}}
    H,R,azel,ex,sysinfo,syscount,vels,vions,vtrps = H_matrix_prl(rs,p,dts,o.data[0].time,nav,sats,exclude)
    return {"status":True,"pos":p,"msg":"success","data":{"residual":resd,'azel':azel,"exclude":ex,"eph":rs,}}


def H_matrix_prl_nlos(satpos,rp,dts,time,nav,sats,nlos,exclude = []):
    ex = [] # exclude sat order
    n = int(len(satpos)/6)
    Ht = []
    Rt = []
    e = prl.Arr1Ddouble(3)
    rr = prl.Arr1Ddouble(3)
    rr[0] = rp[0]
    rr[1] = rp[1]
    rr[2] = rp[2]
    azel = prl.Arr1Ddouble(n*2)
    pos = prl.Arr1Ddouble(3)
    prl.ecef2pos(rr,pos)
    
    dion = prl.Arr1Ddouble(1)
    vion = prl.Arr1Ddouble(1)
    dtrp = prl.Arr1Ddouble(1)
    vtrp = prl.Arr1Ddouble(1)

    vels = []
    vions = []
    vtrps = []

    sysname = prl.Arr1Dchar(4)
    count = {'G':0,'C':0,'E':0,'R':0}
    for i in range(n):
        if i in exclude:
            ex.append(i)
            continue
        sp = np.array(satpos[i*6+0:i*6+3])
        #r = np.linalg.norm(sp-rp[:3])
        r = prl.geodist(satpos[i*6:i*6+3],rr,e)
        azel_tmp = prl.Arr1Ddouble(2)
        prl.satazel(pos,e,azel_tmp)
        azel[i*2] = azel_tmp[0]
        azel[i*2+1] = azel_tmp[1]

        if azel_tmp[1] < 0:
            ex.append(i)
            continue
        

        prl.ionocorr(time,nav,sats[i],pos,azel_tmp,prl.IONOOPT_BRDC,dion,vion)
        prl.tropcorr(time,nav,pos,azel_tmp,prl.TROPOPT_SAAS,dtrp,vtrp)
        vions.append(vion.ptr)
        vtrps.append(vtrp.ptr)
        prl.satno2id(sats[i],sysname)
        try:
            los = nlos[sysname.ptr]
        except:
            los = 0
        if los:
            vel = 0.0001
        else:
            vel = varerr(5*prl.D2R,sysname.ptr[0])
        vels.append(vel)
        if 'G' in sysname.ptr:
            Ht.append([-(sp[0]-rp[0])/r,-(sp[1]-rp[1])/r,-(sp[2]-rp[2])/r,1,0,0,0])
            Rt.append(r+rp[3]-prl.CLIGHT*dts[i*2]+dtrp.ptr+dion.ptr)
            count['G']+=1
        elif 'C' in sysname.ptr:
            Ht.append([-(sp[0]-rp[0])/r,-(sp[1]-rp[1])/r,-(sp[2]-rp[2])/r,0,1,0,0])
            Rt.append(r+rp[4]-prl.CLIGHT*dts[i*2]+dtrp.ptr+dion.ptr)
            count['C']+=1
        elif 'E' in sysname.ptr:
            Ht.append([-(sp[0]-rp[0])/r,-(sp[1]-rp[1])/r,-(sp[2]-rp[2])/r,0,0,1,0])
            Rt.append(r+rp[5]-prl.CLIGHT*dts[i*2]+dtrp.ptr+dion.ptr)
            count['E']+=1
        elif 'R' in sysname.ptr:
            Ht.append([-(sp[0]-rp[0])/r,-(sp[1]-rp[1])/r,-(sp[2]-rp[2])/r,0,0,0,1])
            Rt.append(r+rp[6]-prl.CLIGHT*dts[i*2]+dtrp.ptr+dion.ptr)
            count['R']+=1
    if n-len(ex) > 3:
        H = np.vstack(Ht).astype(np.float64)
        R = np.vstack(Rt).astype(np.float64)
    else:
        H = np.zeros(1)
        R = np.zeros(1)
    sysinfo = np.where(np.any(H!=0,axis=0))[0]
    H = H[:,sysinfo]
    return H,R,azel,ex,sysinfo,count,vels,vions,vtrps

def get_ls_pnt_pos(o,nav,exsatids=[]):
    maxiter = 10
    if o.n < 4:
        return {"status":False,"pos":np.array([0,0,0,0]),"msg":"no enough observations","data":{}}
    rs,noeph,dts,var = get_sat_pos(o.data,o.n,nav)
    w = 1/np.sqrt(np.array(var))
    # if noeph :
    #     print('some sats without ephemeris')
    prs = []
    SNR = []
    exclude = []
    sats = []
    opt = prl.prcopt_default
    skip = 0
    sysname = prl.Arr1Dchar(4)
    vmeas = prl.Arr1Ddouble(1)
    for ii in range(o.n):
        if ii in noeph:
            skip+=1
            continue
        obsd = o.data[ii]
        pii = obsd.P[0] #only process L1
        prl.satno2id(obsd.sat,sysname)
        if  pii == 0 or obsd.sat in exsatids or sysname.ptr[0] not in ['G','C','R','E']:
            exclude.append(ii-skip)
            prs.append(0)
            SNR.append(0)
            sats.append(obsd.sat)
            continue
        pii = prange(obsd,nav,opt,vmeas)
        prs.append(pii)
        sats.append(obsd.sat)
        SNR.append(obsd.SNR[0]/1e3)
        var[ii-skip] += vmeas.ptr
    prs = np.vstack(prs)
    SNR = np.array(SNR)
    p = np.array([0,0,0,0,0,0,0],dtype=np.float64)
    dp = np.array([100,100,100],dtype=np.float64)
    iii = 0 
    var = np.array(var)
    #W = np.eye(o.n-len(exclude))
    b = np.zeros((o.n-len(exclude),1))
    while np.linalg.norm(dp)>0.0001 and iii < maxiter:
        H,R,azel,ex,sysinfo,syscount,vels,vions,vtrps = H_matrix_prl(rs,p,dts,o.data[0].time,nav,sats,exclude)
        inc = set(range(o.n-len(noeph)))-set(ex)
        if len(inc) < 4:
            iii+=1
            continue
        resd = prs[list(inc)] - R
        dp = wls_solve(H,resd)
        p[sysinfo] = p[sysinfo]+dp.squeeze()
        iii+=1
    if iii >= 10:
        return {"status":False,"pos":p,"msg":"over max iteration times","data":{"residual":resd}}
    if np.sqrt((resd*resd).sum()) > 1000:
        return {"status":False,"pos":p,"msg":"residual too large","data":{"residual":resd}}
    H,R,azel,ex,sysinfo,syscount,vels,vions,vtrps = H_matrix_prl(rs,p,dts,o.data[0].time,nav,sats,exclude)
    return {"status":True,"pos":p,"msg":"success","data":{"residual":resd,'azel':azel,"exclude":ex,"eph":rs,'dts':dts,'sats':sats,'prs':prs[list(inc)],'SNR':SNR[list(inc)]}}

if torch_enable:
    def wls_solve_torch(H,z,w = None):
        if w is None:
            w = torch.eye(len(H))
        t1 = torch.matmul(H.T,w)
        t2 = torch.matmul(t1,H)
        t3 = torch.matmul(torch.inverse(t2),H.T)
        t4 = torch.matmul(t3,w)
        x = torch.matmul(t4,z)
        return x

    def get_ls_pnt_pos_torch(o,nav,w = None, p_init = None, exsatids=[]):
        maxiter = 10
        if o.n < 4:
            return {"status":False,"pos":torch.tensor([0,0,0,0],dtype=torch.float32),"msg":"no enough observations","data":{}}
        rs,noeph,dts,var = get_sat_pos(o.data,o.n,nav)
        # if noeph :
        #     print('some sats without ephemeris')
        prs = []
        SNR = []
        exclude = []
        sats = []
        opt = prl.prcopt_default
        skip = 0
        sysname = prl.Arr1Dchar(4)
        vmeas = prl.Arr1Ddouble(1)
        for ii in range(o.n):
            if ii in noeph:
                skip+=1
                continue
            obsd = o.data[ii]
            pii = obsd.P[0] #only process L1
            prl.satno2id(obsd.sat,sysname)
            if  pii == 0 or obsd.sat in exsatids or sysname.ptr[0] not in ['G','C','R','E']:
                exclude.append(ii-skip)
                prs.append(0)
                SNR.append(0)
                sats.append(obsd.sat)
                continue
            pii = prange(obsd,nav,opt,vmeas)
            prs.append(pii)
            sats.append(obsd.sat)
            SNR.append(obsd.SNR[0]/1e3)
            var[ii-skip] += vmeas.ptr
        prs = np.vstack(prs)
        SNR = np.array(SNR)
        if p_init is None:
            p = torch.tensor([0,0,0,0,0,0,0],dtype=torch.float64).to('cuda')
        else:
            p = torch.tensor(p_init,dtype=torch.float64).to('cuda')
        dp = torch.tensor([100,100,100],dtype=torch.float64)
        iii = 0 
        while torch.norm(dp)>0.0001 and iii < maxiter:
            H,R,azel,ex,sysinfo,syscount,vels,vions,vtrps = H_matrix_prl(rs,p.detach().cpu().numpy(),dts,o.data[0].time,nav,sats,exclude)
            inc = set(range(o.n-len(noeph)))-set(ex)
            if len(inc) < 4:
                iii+=1
                continue
            resd = prs[list(inc)] - R
            H = torch.tensor(H,dtype=torch.float64).to('cuda')
            resd = torch.tensor(resd,dtype=torch.float64).to('cuda')
            dp = wls_solve_torch(H,resd,w)
            p[sysinfo] = p[sysinfo]+dp.squeeze()
            iii+=1
        if iii >= 10:
            return {"status":False,"pos":p,"msg":"over max iteration times","data":{"residual":resd}}
        resd = resd.detach().cpu().numpy()
        if np.sqrt((resd*resd).sum()) > 1000:
            return {"status":False,"pos":p,"msg":"residual too large","data":{"residual":resd}}
        H,R,azel,ex,sysinfo,syscount,vels,vions,vtrps = H_matrix_prl(rs,p.detach().cpu().numpy(),dts,o.data[0].time,nav,sats,exclude)
        return {"status":True,"pos":p,"msg":"success","data":{"residual":resd,'azel':azel,"exclude":ex,"eph":rs,'dts':dts,'sats':sats,'prs':prs[list(inc)],'SNR':SNR[list(inc)]}}