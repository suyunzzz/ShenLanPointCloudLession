import numpy as np
import sys
import os
import open3d as o3d

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

def Hash_voxel_filter_random(point_cloud, leaf_size=0.05):
    container_size=100
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
    hashTable=np.zeros((container_size,1),np.float32)  # 哈希容器 0~99

    # 3.计算每一个点所在的box位置
    for i in range(point_cloud.shape[0]):
        hx=(point_cloud[i,0]-x_min)//leaf_size
        hy=(point_cloud[i,1]-y_min)//leaf_size
        hz=(point_cloud[i,2]-z_min)//leaf_size
        h=hx+hy*D_x+hz*D_x*D_y  # 转化为一维数组，这样每一个点都有一个数值代表住在第几号房间

        # 使用HSAH函数映射到0~99
        hash_i = h % 100
        # 先判断该voxel对应的hashTable是否为空
        if hashTable[hash_i,0]==0:  # 如果hashTable为空，则保存当前的voxel索引
            hashTable[h%100]=h
        elif hashTable[hash_i,0]!=0&&h


    return filtered_points

def main():

    # 读取点云文件 txt
    file_name = 'E:\\深蓝学院\\Data\\modelnet40_normal_resampled\\airplane\\airplane_0001.txt'
    points=np.loadtxt(file_name,delimiter=',', dtype=np.float32)
    points=points[:,0:3]
    print('points shape:{}'.format(points.shape))
    print('total points number is:', points.shape[0])
    # 显示原始点云并返回一个open3d的点云类型
    point_cloud_o3d = visualize(points)


    # # 调用voxel滤波函数，实现滤波
    filtered_cloud = Hash_voxel_filter_random(points, 0.1)  # 调用函数
    # filtered_cloud = voxel_filter_mean(points, 0.1)  # 调用函数

    # print('filtered_cloud:{}'.format(filtered_cloud))
    point_cloud_o3d.points = o3d.utility.Vector3dVector(filtered_cloud)  # mat转为pointcloud类型
    # 显示滤波后的点云
    o3d.visualization.draw_geometries([point_cloud_o3d],width=800,height=600)

if __name__ == '__main__':
    main()
