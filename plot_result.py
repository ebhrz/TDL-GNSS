import numpy as np
import matplotlib.pyplot as plt
import sys
import pymap3d as p3d
import pandas as pd

try:
    path = sys.argv[1]
except:
    path = "result/klt1_predict"

try:
    title = sys.argv[2]+" "
except:
    title = ""

item = ['gt','go','rtk','b','w','bw']

labels = ['goGPS','RTKLIB','TDL-B','TDL-W','Ground Truth','TDL-BW']
colors = ['orange', 'brown', 'blue', 'purple','green', 'red']

labels1 = ['goGPS','RTKLIB','TDL-B','TDL-W','TDL-BW']
colors1 = ['orange', 'brown', 'blue', 'purple', 'red']

gtpos = np.loadtxt(path+"/gt.csv",skiprows=1,delimiter=',')
gopos = np.loadtxt(path+"/gogps_pos.csv",skiprows=1,delimiter=',')
rtkpos = np.loadtxt(path+"/rtklib_pos.csv",skiprows=1,delimiter=',')
bpos = np.loadtxt(path+"/TDL_bias_pos.csv",skiprows=1,delimiter=',')
wpos = np.loadtxt(path+"/TDL_weight_pos.csv",skiprows=1,delimiter=',')
bwpos = np.loadtxt(path+"/TDL_bw_pos.csv",skiprows=1,delimiter=',')

gt_init = gopos[0]

errors = []
traj = []

for gt,go,rtk,b,w,bw in zip(gtpos,gopos,rtkpos,bpos,wpos,bwpos):
    errors.append([
        p3d.geodetic2enu(*go,*gt),
        p3d.geodetic2enu(*rtk,*gt),
        p3d.geodetic2enu(*b,*gt),
        p3d.geodetic2enu(*w,*gt),
        p3d.geodetic2enu(*bw,*gt)
    ])
    traj.append([
        p3d.geodetic2enu(*go,*gt_init),
        p3d.geodetic2enu(*rtk,*gt_init),
        p3d.geodetic2enu(*b,*gt_init),
        p3d.geodetic2enu(*w,*gt_init),
        p3d.geodetic2enu(*gt,*gt_init),
        p3d.geodetic2enu(*bw,*gt_init)
    ])

errors = np.array(errors)
traj = np.array(traj)


print("2D error: ",np.linalg.norm(errors[:,:,:2],axis=2).mean(axis=0))
print("3D error: ",np.linalg.norm(errors,axis=2).mean(axis=0))


plt.rcParams.update({'font.size': 30, 'font.family': 'Times New Roman'})

errors3d = {}

# plt.figure(figsize=(16, 10))
# for i in range(5):
#     plt.plot(np.linalg.norm(errors[:,i,:],axis=1),label = labels1[i], color=colors1[i])
#     errors3d[labels1[i]] = np.linalg.norm(errors[:,i,:],axis=1)
# plt.xlim(left=0)
# plt.legend()
# plt.title(title+"3D Error(m)")
# plt.ylabel("Error(m)")
# plt.tight_layout()
# plt.savefig(path+"/error3d.png")

fig, (ax, ax2) = plt.subplots(2, 1, sharex=True, figsize=(16, 10))

# 处理每个标签和对应的颜色
errors3d = {}
for i in range(5):
    norm_data = np.linalg.norm(errors[:, i, :], axis=1)
    errors3d[labels1[i]] = norm_data
    # 绘制到子图
    ax.plot(norm_data, label=labels1[i], color=colors1[i])
    ax2.plot(norm_data, label=labels1[i], color=colors1[i])

# 设置Y轴范围
ax.set_ylim(200, 1200)  # 上图显示较大的错误值
ax2.set_ylim(0, 200)  # 下图显示较小的错误值

# 隐藏两个子图间的间隙
ax.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax.xaxis.tick_top()
ax.tick_params(labeltop=False)
ax2.xaxis.tick_bottom()

# 断裂标记参数
d = 0.015  # 斜杠大小
kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
ax.plot((-d, +d), (-d, +d), **kwargs)
ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)
kwargs.update(transform=ax2.transAxes)
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

# 其他设置
ax.set_title(title + " 3D Error (m)")
ax.set_ylabel("Error (m)")
ax.legend()
plt.tight_layout()

# 保存图片
plt.savefig(path + "/error3d.png")
plt.show()



errors3d = pd.DataFrame(errors3d)
errors3d.to_csv(path+"/errors3d.csv",index=None)


# Redraw the boxplot with Times New Roman font
plt.figure(figsize=(12, 6))
plt.boxplot([errors3d['goGPS'], errors3d['RTKLIB'], errors3d['TDL-B'], errors3d['TDL-W'], errors3d['TDL-BW']],
            labels=['goGPS', 'RTKLIB', 'TDL-B', 'TDL-W', 'TDL-BW'])
plt.title(f'Boxplot of 3D Positioning Errors on {title} Dataset')
plt.ylabel('Error (meters)')
plt.grid(True)
plt.savefig(path+"/error3dbox.png")


plt.figure(figsize=(16, 10))
for i in range(5):
    plt.plot(np.linalg.norm(errors[:,i,:2],axis=1),label = labels1[i], color=colors1[i])
plt.xlim(left=0)
plt.legend()
plt.title(title+"2D Error(m)")
plt.ylabel("Error(m)")
plt.tight_layout()
plt.savefig(path+"/error2d.png")


plt.figure(figsize=(10, 10))
for i in range(6):
    plt.plot(traj[:,i,0],traj[:,i,1], marker='o', linestyle='-', color=colors[i],label=labels[i],markersize=5)
handles, labels = plt.gca().get_legend_handles_labels()
order = [4, 0, 1, 2,3,5]  # 指定图例的顺序：line3, line1, line2
plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])
plt.title(title+"2D Trajectory(m)")
plt.axis('equal')
plt.tight_layout()
plt.savefig(path+"/traj2d.png")
plt.show()
