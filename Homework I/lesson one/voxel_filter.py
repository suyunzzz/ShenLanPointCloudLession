# 实现voxel滤波，并加载数据集中的文件进行验证

import open3d as o3d 
import os
import numpy as np
from pyntcloud import PyntCloud

# 使用open3d可视化点云,返回pointcloud类型示例
# 输入为一个矩阵n*3
def visualize(pointcloud):
    from open3d.open3d.geometry import PointCloud
    from open3d.open3d.utility import Vector3dVector
    from open3d.open3d.visualization import draw_geometries

    # from open3d_study import *


    # points = np.random.rand(10000, 3)
    point_cloud = PointCloud()
    point_cloud.points = Vector3dVector(pointcloud[:,0:3].reshape(-1,3))
    draw_geometries([point_cloud],width=800,height=600)
    return point_cloud

# 功能：对点云进行voxel滤波
# 输入：
#     point_cloud：输入点云n*3
#     leaf_size: voxel尺寸
def voxel_filter_mean(point_cloud, leaf_size=0.05):
    pt=np.zeros([point_cloud.shape[0],4],dtype=np.float32)  # 创建一个矩阵，前三列放点，第四列放房间号
    pt[:,:-1]=point_cloud


    # 作业3
    # 屏蔽开始
    # 1.找到点云的最大值，最小值
    max=point_cloud.max(axis=0)
    min=point_cloud.min(axis=0)
    x_max=max[0]
    y_max=max[1]
    z_max=max[2]
    x_min=min[0]
    y_min=min[1]
    z_min=min[2]
    print('max:{},min:{}'.format(max,min))

    # 2.计算每一个维度上的box的个数
    D_x=(x_max-x_min)//leaf_size+1  # 地板除
    D_y=(y_max-y_min)//leaf_size+1
    D_z=(z_max-z_min)//leaf_size+1

    # 3.计算每一个点所在的box位置
    h=np.zeros((point_cloud.shape[0],1),np.float32)
    for i in range(point_cloud.shape[0]):
        hx=(point_cloud[i,0]-x_min)//leaf_size
        hy=(point_cloud[i,1]-y_min)//leaf_size
        hz=(point_cloud[i,2]-z_min)//leaf_size
        h_=hx+hy*D_x+hz*D_x*D_y  # 转化为一维数组，这样每一个点都有一个数值代表住在第几号房间
        h[i,0]=h_  # 保存每一个点的房间号

    h=h.reshape(-1)
    print('h :{}'.format(h.shape))  # h :(10000,)

# 4.根据房间号对点进行排序
    index=np.argsort(h).tolist()
    print('index:{}'.format(index))
    h=h[index].tolist()  # 与index保持一致
    print('h:{}'.format(h))

# 创建一个字典 {voxel1:[],voxel2:[],....}
    dict={}
    for voxel,point in zip(h,index):
        if voxel not in dict.keys():
            dict[voxel]=point
        else:
            if isinstance(dict[voxel],list):
                dict[voxel].append(point)
            else:
                dict[voxel]=[dict[voxel]]
                dict[voxel].append(point)
    print('dict length:{},dict:{}'.format(len(dict), dict))

    # 对每一个voxel进行采样
    filter_points=[]
    for item in dict.items():
        if isinstance(item[1],list):
            point_mean=np.mean(point_cloud[item[1]],axis=0)
            filter_points.append(point_mean)
        else:
            filter_points.append(point_cloud[item[1]])

    # 屏蔽结束

    # 把点云格式改成array，并对外返回
    filter_points = np.array(filter_points, dtype=np.float64)
    print('filter_points:{}'.format(filter_points.shape))
    return filter_points

def voxel_filter_random(point_cloud, leaf_size=0.05):
    pt=np.zeros([point_cloud.shape[0],4],dtype=np.float32)  # 创建一个矩阵，前三列放点，第四列放房间号
    pt[:,:-1]=point_cloud

    filtered_points = []  # 创建一个list
    # 作业3
    # 屏蔽开始
    # 1.找到点云的最大值，最小值
    max=point_cloud.max(axis=0)
    min=point_cloud.min(axis=0)
    x_max=max[0]
    y_max=max[1]
    z_max=max[2]
    x_min=min[0]
    y_min=min[1]
    z_min=min[2]
    print('max:{},min:{}'.format(max,min))

    # 2.计算每一个维度上的box的个数
    D_x=(x_max-x_min)//leaf_size+1  # 地板除
    D_y=(y_max-y_min)//leaf_size+1
    D_z=(z_max-z_min)//leaf_size+1

    # 3.计算每一个点所在的box位置
    for i in range(point_cloud.shape[0]):
        hx=(point_cloud[i,0]-x_min)//leaf_size
        hy=(point_cloud[i,1]-y_min)//leaf_size
        hz=(point_cloud[i,2]-z_min)//leaf_size
        h=hx+hy*D_x+hz*D_x*D_y  # 转化为一维数组，这样每一个点都有一个数值代表住在第几号房间
        pt[i,-1]=h  # 保存每一个点的房间号
    # 4.根据房间号对点进行排序
    # print('pt[:,-1]:{}'.format(pt[:,-1]))  # 1000*1
    index=np.argsort(pt[:,-1]).tolist()  # 对房间号进行排序
    print('点的index:{}'.format(index))
    pt[:,-1]=pt[:,-1][index]
    print('pt[:,-1]:{}'.format(pt[:,-1]))
    h=pt[:,-1].tolist()
    print('h:{}'.format(h))

    dict={}  # {voxel1:[point1,point2,...],...}
    for voxel,point in zip(h,index):
        if voxel not in dict.keys():
            dict[voxel]=point
        else:
            if isinstance(dict[voxel],list):
                dict[voxel].append(point)
            else:
                dict[voxel]=[dict[voxel]]
                dict[voxel].append(point)
    print('dict length:{},dict:{}'.format(len(dict),dict))

    # 遍历每一个voxel
    index_select=[]
    for item in dict.items():
        if isinstance(item[1],list):
            index_select.append(np.random.choice(item[1]))
        else:
            index_select.append(item[1])

    print('index_select :{}'.format(index_select))

    filtered_points=point_cloud[index_select]

    print('random filtered_points num :{}'.format(len(filtered_points))) # 降采样后点的数目
    # print('random filtered_points:{}'.format(filtered_points))
    # 把点云格式改成array，并对外返回
    filtered_points = np.array(filtered_points, dtype=np.float64)
    # print('filtered_points:{}'.format(filtered_points))
    return filtered_points


def main():

    # 读取点云文件 txt
    file_name = 'F:\\深蓝学院\\Data\\modelnet40_normal_resampled\\airplane\\airplane_0001.txt'
    points=np.loadtxt(file_name,delimiter=',', dtype=np.float32)
    points=points[:,0:3]
    print('points shape:{}'.format(points.shape))
    print('total points number is:', points.shape[0])
    # 显示原始点云并返回一个open3d的点云类型
    point_cloud_o3d = visualize(points)


    # # 调用voxel滤波函数，实现滤波
    # filtered_cloud = voxel_filter_random(points, 0.1)  # 调用函数
    filtered_cloud = voxel_filter_mean(points, 0.1)  # 调用函数


    point_cloud_o3d.points = o3d.utility.Vector3dVector(filtered_cloud)  # mat转为pointcloud类型
    # 显示滤波后的点云
    o3d.visualization.draw_geometries([point_cloud_o3d],width=800,height=600)

if __name__ == '__main__':
    main()
