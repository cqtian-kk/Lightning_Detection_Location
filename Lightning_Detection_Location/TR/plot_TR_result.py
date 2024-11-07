#%% 
import numpy as np
import matplotlib.pyplot as plt

# 原始数据数量
num_events = 271
filtered_events = [False] * num_events  # 初始化筛选标志数组

sample = 0.0000002  #采样间隔s
#各台站位置
sta_x = [-3.324113012, 14.97894469, 1.252397509, -11.96347912, -5.929079939, 4.98532987]
sta_y = [-0.029319449, 4.206803539, 11.66024498,4.639923346, -13.80897884, -6.668673572]
sta_z = [0.015169, 0.020862, 0.030253, 0.033826, 0.037787244, 0.033646]
stationlist = []
for i in range(len(sta_x)):
    stationlist.append([sta_x[i], sta_y[i], sta_z[i]])
stationlist = np.array(stationlist)
# 读取波形数据
# data = np.loadtxt('./data/EMD_400-500ns.txt', skiprows=1)
result = []

latgrid = 1
longrid = 1
highgrid = 1
studyarea = [-20, -20, 41, 41]  # km
studydepth = [0, 15]  # km
lats = np.arange(studyarea[0],  studyarea[0]+studyarea[2], latgrid)
lons = np.arange(studyarea[1],  studyarea[1]+studyarea[3], longrid)
highs = np.arange(studydepth[0], studydepth[1], highgrid)

data = np.loadtxt('../GNNs/gnnresult/VHFresult.txt')
data1 = np.load('./Location_result/TR271_10.npz')

data1s = data1['d2']
data1times = data1['d1']
# 提取数据点坐标和颜色
x1 = [d[1] for d in data]
y1 = [d[2] for d in data]
z1 = [d[3] for d in data]
colors1 = [d[0]*1000 for d in data]
x = [d[0] for d in data1s]
y = [d[1] for d in data1s]
z = [d[2] for d in data1s]
colors = [d for d in data1times]

# 用于计算空间距离的函数
def compute_distances(event1, event2):
    return np.linalg.norm(np.array(event1) - np.array(event2))

# 存储新的三维坐标点和对应时间
new_data_points = []
new_data_times = []

# 遍历每个事件
for i, (xi, yi, zi) in enumerate(zip(x, y, z)):
    distances_with_time = []
    distances2d_with_time = []
    # 计算当前事件与其他事件的空间距离以及时间距离
    for j, (xj, yj, zj) in enumerate(zip(x, y, z)):
        if i != j:  # 跳过自身
            distance = compute_distances((xi, yi, zi), (xj, yj, zj))
            distance2d = compute_distances((xi, yi), (xj, yj))
            time_distance = abs(colors[i] - colors[j])  # 时间距离
            distances_with_time.append((distance, time_distance))
            distances2d_with_time.append((distance2d, time_distance))
    # 找到时间上临近的三个事件
    nearest_events = sorted(distances_with_time, key=lambda x: x[1])[:4]
    min_distances = [event[0] for event in nearest_events]
    min_distance = min(min_distances)
    nearest_events2d = sorted(distances2d_with_time, key=lambda x: x[1])[:4]
    min_distances2d = [event[0] for event in nearest_events2d]
    min_distance2d = min(min_distances2d)
    if min_distance <= 7 and min_distance2d <= 4.5:  # 如果最近距离小于等于5，保留该点
        new_data_points.append((xi, yi, zi))
        new_data_times.append(colors[i])
        filtered_events[i] = True  # 标记该事件为保留

# 输出筛选标志数组
filtered_events = np.array(filtered_events)


# 将新的数据点和时间转换为NumPy数组
new_data_points = np.array(new_data_points)
new_data_times = np.array(new_data_times)

x = [d[0] for d in new_data_points]
y = [d[1] for d in new_data_points]
z = [d[2] for d in new_data_points]
colors = [d for d in new_data_times]

fig = plt.figure(dpi = 1500)

fig.subplots_adjust(hspace=0.0)
fig.subplots_adjust(wspace=0.0)
gs0 = fig.add_gridspec(nrows=20,ncols=18,left=0.05,right=0.45) 
ax1 = fig.add_subplot(gs0[5:9,0:12])
ax2 = fig.add_subplot(gs0[9:18,0:12])
ax3 = fig.add_subplot(gs0[9:18,13:18])
ax4 = fig.add_subplot(gs0[0:4, 0:18])
ax5 = fig.add_subplot(gs0[5:9, 13:18])
plt.subplots_adjust(hspace=3, wspace=3)
xlimRange = [studyarea[0],studyarea[0]+studyarea[2]]
ylimRange = [studyarea[1],studyarea[1]+studyarea[3]]
zlimRange = [studydepth[0],studydepth[1]]
        
#-- yoz slice
ax1.scatter( x, z, marker="s", s=0.1, c=colors, zorder=101, cmap = 'jet' )
ax1.scatter( x1, z1, marker="s", s=0.1, c='0.88', zorder=100 )

ax1.set_ylabel('Height (km)',labelpad = 1, fontsize=6, fontweight='bold')
ax1.set_yticks(np.arange( 0,15,step = 5))
ax1.set_xticks(np.arange( -20, 20,step = 5))
ax1.set_xlim( xlimRange )
ax1.set_ylim( zlimRange )
# 设置x轴和y轴的刻度线向内移动
ax1.tick_params(axis='x', direction='in', pad=2, labelsize=6, length=2)
ax1.tick_params(axis='y', direction='in', pad=2, labelsize=6, length=2)
for label in ax1.get_xticklabels():
    label.set_weight('bold')
for label in ax1.get_yticklabels():
    label.set_weight('bold')
ax1.text(-19.5,12,s = 'II)',fontsize=8, weight='bold')
ax1.set_aspect('auto')
# ax1.grid()

#-- xoy slice
ax2.scatter( x,y, marker="s", s=0.1, c=colors, zorder=101, cmap = 'jet'  )
ax2.scatter( x1, y1, marker="s", s=0.1, c='0.88', zorder=100 )
ax2.set_xticks(np.arange( -20,20,step = 5))
ax2.set_yticks(np.arange( -20,20,step = 5))
ax2.set_ylabel('South-North(km)',labelpad = 1, fontsize=6, fontweight='bold')
ax2.set_xlabel('West-East (km)',labelpad = 1, fontsize=6, fontweight='bold')
ax2.set_xlim( xlimRange )
ax2.set_ylim( ylimRange )
# 设置x轴和y轴的刻度线向内移动
ax2.tick_params(axis='x', direction='in', pad=2, labelsize=6, length=2)
ax2.tick_params(axis='y', direction='in', pad=2, labelsize=6, length=2)
for label in ax2.get_xticklabels():
    label.set_weight('bold')
for label in ax2.get_yticklabels():
    label.set_weight('bold')
ax2.text(-19.5,18,s = 'IV)',fontsize=8, weight='bold')
ax2.set_aspect('auto')
# ax2.grid()
if len(stationlist) != 0:
    for sta in stationlist:
        ax2.scatter( sta[1], sta[0], marker="^", s=10, color='black', zorder=100 )
#-- xoz slice
ax3.scatter( z, y, marker="s", s=0.1, c=colors, zorder=101, cmap = 'jet' )
ax3.scatter( z1,y1, marker="s", s=0.1, c='0.88', zorder=100)
ax3.set_xlabel('Height (km)',labelpad = 1, fontsize=6, fontweight='bold')
ax3.set_yticks(np.arange( -20,20,step = 5))
ax3.set_xticks(np.arange( 0,15,step = 5))
ax3.set_xlim( zlimRange )
ax3.set_ylim( ylimRange )
# 设置x轴和y轴的刻度线向内移动
ax3.tick_params(axis='x', direction='in', pad=2, labelsize=6, length=2)
ax3.tick_params(axis='y', direction='in', pad=2, labelsize=6, length=2)
for label in ax3.get_xticklabels():
    label.set_weight('bold')
for label in ax3.get_yticklabels():
    label.set_weight('bold')
ax3.set_aspect('auto')
ax3.text(0.5,18,s = 'V)',fontsize=8, weight='bold')
# ax3.grid()

ax4.scatter( colors, z, marker="s", s=0.1, c=colors, zorder=101, cmap = 'jet' )
ax4.scatter( colors1, z1, marker="s", s=0.1, c='0.88', zorder=100 )
ax4.set_ylabel('Height (km)',labelpad = 1, fontsize=6, fontweight='bold')
ax4.set_xlabel('Times (ms)',labelpad = 1, fontsize=6, fontweight='bold')
ax4.set_xticks(np.arange( 400, 1000.1,step = 100))
ax4.set_yticks(np.arange( 0, 15,step = 5))
ax4.set_ylim( zlimRange )
# 设置x轴和y轴的刻度线向内移动
ax4.tick_params(axis='x', direction='in', pad=2, labelsize=6, length=2)
ax4.tick_params(axis='y', direction='in', pad=2, labelsize=6, length=2)
for label in ax4.get_xticklabels():
    label.set_weight('bold')
for label in ax4.get_yticklabels():
    label.set_weight('bold')
ax4.text(408.5,12,s = 'I)',fontsize=8, weight='bold')
ax4.set_aspect('auto')

# ax4.grid()

# 计算每个区间的计数
counts, bins = np.histogram(z, bins=25, range=(0, 15))
ax5.plot(counts,bins[1:],c ='grey',linewidth=1)
ax5.set_xticks(np.arange(0, 60,step = 20))
ax5.set_yticks(np.arange( 0, 15,step = 5))
# 设置x轴和y轴的刻度线向内移动
ax5.tick_params(axis='x', direction='in', pad=2, labelsize=6, length=2)
ax5.tick_params(axis='y', direction='in', pad=2, labelsize=6, length=2)
for label in ax5.get_xticklabels():
    label.set_weight('bold')
for label in ax5.get_yticklabels():
    label.set_weight('bold')
ax5.set_aspect('auto')
ax5.text(8,2,s = str(len(new_data_points))+' Points',fontsize=5,  weight='bold')
ax5.text(4,13,s = 'III)',fontsize=8, weight='bold')
fig.suptitle('2018-06-27 23:56:13 LF TR Mapping',x=0.11, y=0.91, ha='left', va='top', fontsize=6, fontweight='bold')
# # 调整子图间距
# plt.tight_layout()
# 显示图形
plt.show()
# %%
