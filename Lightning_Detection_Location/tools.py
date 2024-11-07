#%%
import numpy as np
from numba import jit
import numpy
import math
import matplotlib.pyplot as plt
import numpy.fft as fft

def calculate_travel_times(studygrids, stationlist, c):
    '''
    计算走时表
    studygrids = (n, 3), stationlist = (m, 3)
    travel_time = (n, 4)
    '''
    # 利用广播计算所有格点到所有站点的距离差
    diff = studygrids[:, np.newaxis, :] - stationlist[np.newaxis, :, :]
    
    # 计算欧氏距离，不需要平方和开方的中间步骤
    dist_squared = np.sum(diff**2, axis=-1)
    
    # 计算走时
    travel_time = np.sqrt(dist_squared) / c
    
    return travel_time

def creategrid(lats, lons, highs):
    '''
    生产网格点
    '''
    # Generate grid matrices for each dimension
    lat_grid, lon_grid, high_grid = np.meshgrid(lats, lons, highs, indexing='ij')
    
    # Flatten the grids and stack them into a single 2D array of shape (N, 3)
    studygrids = np.stack([lat_grid.ravel(), lon_grid.ravel(), high_grid.ravel()], axis=-1)
    
    return studygrids

def normalization(data, axis=None):
    '''
    对data进行归一化0-1的操作,
    同时兼容一维和二维data,默认对行进行归一化。
    当一个或多个行/列的范围(_range)为0时，返回原始的行/列值。
    '''
    # 如果数据是一维的，强制axis为0
    if data.ndim == 1:
        axis = 0
    elif axis is None:
        # 默认对二维数组的行进行归一化
        axis = 1

    min_val = np.min(data, axis=axis, keepdims=True)
    max_val = np.max(data, axis=axis, keepdims=True)
    _range = max_val - min_val

    # 归一化处理，其中 _range 为 0 的部分返回原始数据
    normalized_data = np.where(_range == 0, data, (data - min_val) / _range)
    return normalized_data
def StandardDN(data, axis=None):
    '''
    对data进行标准差归一化操作 (Z-score normalization)，
    同时兼容一维和二维data，默认对行进行归一化。
    '''
    # 如果数据是一维的，强制axis为0
    if data.ndim == 1:
        axis = 0
    elif axis is None:
        # 默认对二维数组的行进行归一化
        axis = 1

    mean_val = np.mean(data, axis=axis, keepdims=True)
    std_dev = np.std(data, axis=axis, keepdims=True)

    # 避免除以零的情况，对标准差为0的情况进行处理
    std_dev[std_dev == 0] = 1  # 避免除以零

    standardized_data = (data - mean_val) / std_dev
    return standardized_data

       


def plotresult3Dcube(data_list,studyarea,studydepth):
    '''
    #数据示例data_list = [[1, [1, 2, 3]], [2, [4, 5, 6]], [3, [7, 8, 9]]]
    '''
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    
    # 提取数据点坐标和颜色
    x = [d[1][0] for d in data_list]
    y = [d[1][1] for d in data_list]
    z = [d[1][2] for d in data_list]
    colors = [d[0] for d in data_list]
    
    # 初始化三维图形
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制散点图，并指定每个点的颜色
    scatter = ax.scatter(x, y, z, c=colors)
    
    ax.set_xlim(studyarea[0], studyarea[0]+studyarea[2])
    ax.set_ylim(studyarea[1], studyarea[1]+studyarea[3])
    ax.set_zlim(studydepth[0], studydepth[1])
    # 添加颜色条
    sm = ScalarMappable(cmap=scatter.cmap)
    sm.set_array(colors)
    cbar = plt.colorbar(sm)

    # 设置颜色条标签
    cbar.ax.set_ylabel('Data Value')

    # 显示图形
    plt.show()
    

def plotresult3Dslices(data_list,studyarea,studydepth,stationlist = []):
    '''
    使用示例：假设数据存在于data_list中
    studyarea = [0, 1, 8, 8]  # km
    studydepth = [2, 10]  # km
    data_list = [[1, [1, 2, 3]], [2.5, [4, 6, 6]],[2.7, [4, 5, 6.1]], [3, [7, 8, 9]]]
    '''
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    
    
    # 提取数据点坐标和颜色
    x = [d[1][0] for d in data_list]
    y = [d[1][1] for d in data_list]
    z = [d[1][2] for d in data_list]
    colors = [d[0] for d in data_list]
    

    fig = plt.figure()
    fig.subplots_adjust(hspace=0.0)
    fig.subplots_adjust(wspace=0.0)
    gs0 = fig.add_gridspec(nrows=12,ncols=14,left=0.05,right=0.48,wspace=0.4) 
    ax1 = fig.add_subplot(gs0[3:5,0:8])
    ax2 = fig.add_subplot(gs0[6:12,0:8])
    ax3 = fig.add_subplot(gs0[6:12,9:12])
    ax4 = fig.add_subplot(gs0[0:2, 0:12])
    

    xlimRange = [studyarea[0],studyarea[0]+studyarea[2]]
    ylimRange = [studyarea[1],studyarea[1]+studyarea[3]]
    zlimRange = studydepth
            
    #-- yoz slice
    ax1.scatter( x, z, marker=".", s=5, c=colors, zorder=101 )
    ax1.set_ylabel('Depth (km)')
    ax1.set_xlabel('x(km)')
    ax1.set_yticks(np.arange( 0,15,step = 5))
    ax1.set_xticks(np.arange( -20, 20,step = 5))
    ax1.set_xlim( xlimRange )
    ax1.set_ylim( zlimRange )
    ax1.set_aspect('auto')
    ax1.grid()
    
    #-- xoy slice
    ax2.scatter( x,y, marker=".", s=5, c=colors, zorder=101 )
    ax2.set_xticks(np.arange( -20,20,step = 5))
    ax2.set_yticks(np.arange( -20,20,step = 5))
    ax2.set_ylabel('y (km)')
    ax2.set_xlabel('x(km)')
    ax2.set_xlim( xlimRange )
    ax2.set_ylim( ylimRange )
    ax2.set_aspect('auto')
    ax2.grid()
    if len(stationlist) != 0:
        for sta in stationlist:
            ax2.scatter( sta[1], sta[0], marker="^", s=25, color='black', zorder=100 )
    #-- xoz slice
    ax3.scatter( z, y, marker=".", s=5, c=colors, zorder=101 )
    ax3.set_ylabel('y(km)')
    ax3.set_xlabel('Depth (km)')
    ax3.set_yticks(np.arange( -20,20,step = 5))
    ax3.set_xticks(np.arange( 0,15,step = 5))
    ax3.set_xlim( zlimRange )
    ax3.set_ylim( ylimRange )
    ax3.set_aspect('auto')
    ax3.grid()
    
    ax4.scatter( colors, z, marker=".", s=5, c=colors, zorder=101 )
    ax4.set_xlabel('time')
    ax4.set_ylabel('Depth (km)')
    ax4.set_xticks(np.arange( np.min(colors), np.max(colors),step = 100))
    ax4.set_yticks(np.arange( studydepth[0], studydepth[1],step = 5))
    ax4.set_aspect('auto')
    ax4.grid()
    # plt.tight_layout()
    # cax = fig.add_axes([1, 0.1, 0.02, 0.8])  # left, bottom, width, height
    # cbar = fig.colorbar(imageXOY, cax=cax)
    # # 添加颜色条
    # sm = ScalarMappable(cmap=scatter.cmap)
    # sm.set_array(colors)
    # cbar = plt.colorbar(sm)
    # # 设置颜色条标签
    # cbar.ax.set_ylabel('Data Value')
    
    # 显示图形
    plt.show()
    
def plotresult3Dslices_compa(stationlist, VHFresult, result_raw):
    '''
    用于画两个3D定位点的mapping,VHFresult是参考点,result_raw是定位点，
    同时对定位点根据其时间连续性进行过滤
    stationlist为列表
    VHFresult,result_raw为文件路径命, 
    VHFresult对应格式txt, 
    result_raw对应格式npz, d1是三维坐标,d2是时间(在图中用颜色表示)
    '''

    result = []

    data = np.loadtxt(VHFresult)
    data1 = np.load(result_raw)
    # data1 = np.load('./gnnresult/resultEMD3-500TR_89.npz')
    data1s = data1['d1']
    data1times = data1['d2']
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
        # 找到时间上临近的20个事件
        # nearest_events = sorted(distances_with_time, key=lambda x: x[1])[:20]
        # distances = [event[0] for event in nearest_events]
        # distances = np.array(distances)
        # CR = np.sum(distances<10)/20
        # if CR > 0.2 :  # 如果最近距离小于等于5，保留该点
        #     new_data_points.append((xi, yi, zi))
        #     new_data_times.append(colors[i])

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
    ax1.text(-19.5,12,s = 'b)',fontsize=8, weight='bold')
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
    ax2.text(-19.5,18,s = 'd)',fontsize=8, weight='bold')
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
    ax3.text(0.5,18,s = 'e)',fontsize=8, weight='bold')
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
    ax4.text(408.5,12,s = 'a)',fontsize=8, weight='bold')
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
    ax5.text(4,13,s = 'c)',fontsize=8, weight='bold')
    fig.suptitle('2018-06-27 23:56:13 LF GNN Mapping',x=0.11, y=0.91, ha='left', va='top', fontsize=6, fontweight='bold')
    # # 调整子图间距
    # plt.tight_layout()
    # 显示图形
    plt.show()
    # plt.savefig('gnnpng/285.png')

def TRShot3d(lats, lons, highs, event, stationlist, E ):
    '''
    lats, lons, highs为组成E的三个维度的数组
    stationlist为台站列表
    显示能量数组E在event这个点的三维切片图
    '''
    # 计算数据的最小值和最大值
    data_min = np.nanmin(E)
    data_max = np.nanmax(E)
    latnum,lonnum,highnum = len(lats),len(lons),len(highs)
    latgrid,longrid,highgrid = lats[1]-lats[0],lons[1]-lons[0],highs[1]-highs[0]
    studydepth = [highs[0],highs[-1]+highgrid]
    snapShot3dPS = np.zeros([ latnum, lonnum, highnum ])
    for i in range( latnum ):
        for j in range( lonnum ):
            for k in range( highnum ):
                index = i*lonnum*highnum + j*highnum + k
                snapShot3dPS[i, j, k] = E[index]
    
    srcXdegIdx = np.int( np.floor( (event[0]-lats[0])/latgrid) )
    srcYdegIdx = np.int( np.floor( (event[1]-lons[0])/longrid) )
    srcHdegIdx = np.int( np.floor( (event[2]-highs[0])/highgrid) )
    
    # print('srcXdegIdx = ', srcXdegIdx)
    # print('srcYdegIdx = ', srcYdegIdx)
    # print('srcHdegIdx = ', srcHdegIdx)
                      
    snapShot2dYOZ = snapShot3dPS[:,srcYdegIdx,:]/np.max(snapShot3dPS[srcXdegIdx,:,:])
    snapShot2dXOY = snapShot3dPS[:,:,srcHdegIdx]/np.max(snapShot3dPS[:,:,srcHdegIdx])
    snapShot2dXOZ = snapShot3dPS[srcXdegIdx,:,:]/np.max(snapShot3dPS[:,srcYdegIdx,:])
    # snapShot2dYOZ = snapShot3dPS[:,srcYdegIdx,:]
    # snapShot2dXOY = snapShot3dPS[:,:,srcHdegIdx]
    # snapShot2dXOZ = snapShot3dPS[srcXdegIdx,:,:]
    shape_yoz = np.shape(snapShot2dYOZ)
    shape_xoy = np.shape(snapShot2dXOY)
    shape_xoz = np.shape(snapShot2dXOZ)
    # print( 'shape_yoz =', shape_yoz)
    # print( 'shape_xoy =', shape_xoy)
    # print( 'shape_xoz =', shape_xoz)
    
    fig = plt.figure( figsize=(6,6),dpi = 1500)
    fig.subplots_adjust(hspace=0.0)
    fig.subplots_adjust(wspace=0.0)
    gs0 = fig.add_gridspec( 9, 9 )
    ax1 = fig.add_subplot(gs0[0:5, 0:3])
    ax2 = fig.add_subplot(gs0[0:5, 4:9])
    ax3 = fig.add_subplot(gs0[6:9, 4:9])
    
    minLat,maxLat,minLon,maxLon = lats[0],lats[-1],lons[0],lons[-1]
    xlimRange = [minLon,maxLon]
    ylimRange = [minLat,maxLat]
            
    #-- yoz slice
    extent=[ studydepth[0], studydepth[1], minLat, maxLat  ]
    imageYOZ = ax1.imshow( snapShot2dYOZ, interpolation='none',
                         cmap='rainbow', extent=extent, origin='lower')
    # ax1.scatter( source_O[2], source_O[0], marker="*", s=50, color='black', zorder=101 )
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax1.set_xlabel('Depth (km)', fontsize=12)
    ax1.set_ylabel('Latitude ($^\circ$)', fontsize=12)
    ax1.set_xticks(np.arange( studydepth[0], studydepth[1], step=4))
    ax1.set_yticks(np.arange( minLat, maxLat, step=4 ))
    ax1.set_xlim(ax1.get_xlim()[::-1])
    ax1.set_aspect('auto')
    ax1.grid()
    
    #-- xoy slice
    extent=[ minLon, maxLon, minLat, maxLat]
    imageXOY = ax2.imshow( snapShot2dXOY, interpolation='none',
                         cmap='rainbow', extent=extent, origin='lower')
    # # 添加右边占满整张图片高度的坐标轴
    # cax = fig.add_axes([1, 0.1, 0.02, 0.8])  # left, bottom, width, height
    # cbar = fig.colorbar(imageXOY, cax=cax)
    
    for sta in stationlist:
        ax2.scatter( sta[1], sta[0], marker="^", s=25, color='black', zorder=100 )
    # ax2.scatter( source_O[1], source_O[0], marker="*", s=50, color='black', zorder=101 )
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_xlim( xlimRange )
    ax2.set_ylim( ylimRange )
    ax2.set_aspect('auto')
    ax2.grid()

    #-- xoz slice
    extent=[ minLon, maxLon, studydepth[0], studydepth[1] ]
    imageXOZ= ax3.imshow( snapShot2dXOZ.T, interpolation='none',
                         cmap='rainbow',  extent=extent, origin='lower')
    # ax3.scatter( source_O[1], source_O[2], marker="*", s=50, color='black', zorder=101 )
    ax3.tick_params(axis='both', which='major', labelsize=12)
    ax3.set_xlabel('Longitude ($^\circ$)', fontsize=12)
    ax3.set_ylabel('Depth (km)', fontsize=12)
    ax3.set_xticks(np.arange( minLon, maxLon, step=4 ))
    ax3.set_yticks(np.arange( highs[0], highs[-1], step=4))
    ax3.set_ylim(ax3.get_ylim()[::-1])
    ax3.set_aspect('auto')
    ax3.grid()
    plt.tight_layout()
    plt.show()


def plotvalue(values, times=None, save_plot=False, png_name='./waveform.png', 
    axis_off=False, color='black', dpi = 500):
    '''
    values要求必须是二维数组(n,m)且是必要的,
    times是对应大小m的数组,图的x轴坐标,默认为数据个数,
    save_plot, png_name是保存图件时的参数,默认不保存,
    axis_off控制图坐标的显示,默认显示,
    color是控制图的颜色,可以是颜色列表或者一个颜色,默认全为黑色
    dpi是图的清晰度
    '''
    n, m = values.shape
    fig_height = n
    fig_width = max(6, m / 100)
    fig, axes = plt.subplots(n, 1, figsize=(fig_width, fig_height), dpi = dpi)
    
    # 默认时间轴处理
    if times is None:
        times = np.linspace(0, m - 1, m)
    
    # 确定颜色列表
    if isinstance(color, str):
        # 单一颜色，为所有波形使用这一颜色
        colors = [color] * n
    elif len(color) < n:
        # 颜色列表不足，循环使用颜色
        colors = [color[i % len(color)] for i in range(n)]
    else:
        # 颜色列表足够或更多
        colors = color

    # 绘图
    for i, ax in enumerate(axes):
        ax.plot(times, values[i], c=colors[i])
        ax.text(0.02, 0.92, f"Waveform {i + 1}", transform=ax.transAxes, verticalalignment='top', horizontalalignment='left')
        if axis_off:
            ax.axis('off')
    if not axis_off:
        # 设置整个图的x和y轴标签
        fig.text(0.5, 0.04, 'Time', ha='center', va='center')  # X轴标签
        fig.text(0.09, 0.5, 'Amplitude', ha='center', va='center', rotation='vertical')  # Y轴标签

    # 如果指定了保存图像
    if save_plot:
        plt.savefig(png_name)

    plt.show()



# %%