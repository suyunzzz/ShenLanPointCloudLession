# 实现voxel滤波，并加载数据集中的文件进行验证

import open3d as o3d 
import os
import numpy as np
from pyntcloud import PyntCloud

# 使用open3d可视化点云,返回pointcloud类型示例
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
    index=pt[:,-1].argsort()  # 对房间号进行排序
    print('index:{}'.format(index))
    pt=pt[index]
    print('pt:{}'.format(pt))

    # 5.遍历每一个房间，随机选择一个点
    room_num=D_x*D_y*D_z
    print('room_num:{}'.format(room_num))  # 房间的总数
    for room in range(int(room_num)): # 遍历所有房间
        temp=[]
        p_num=0  # 统计房间内点的数目
        # print('room:{}'.format(room))
        for i in range(point_cloud.shape[0]): # 遍历所有点
            if pt[i,-1]==room:  # 如果这个点在房间内,累加，并查询下一个点
                p_num+=1  # 房间内点的+1
                temp.append(pt[i,:-1])
        if p_num!=0: # 如果房间有点 才有接下来的计算
            temp_p=np.array(temp,dtype=np.float32)  # 应该是一个M*3的矩阵  M为该房间点的个数
            p_mean=np.mean(temp_p,axis=0)  # 房间内的平均值  1*3的矩阵

            filtered_points.append(p_mean)  # 保存每一个房间的平均值

        if room*100/room_num%10==0:
            print('完成进度：{}%'.format(room*100/room_num))

    print('filtered_points num :{}'.format(len(filtered_points))) # 降采样后点的数目

    # 屏蔽结束

    # 把点云格式改成array，并对外返回
    filtered_points = np.array(filtered_points, dtype=np.float64)
    return filtered_points

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
    index=pt[:,-1].argsort()  # 对房间号进行排序
    print('点的index:{}'.format(index))
    pt=pt[index]
    print('pt:{}'.format(pt))

    # 5.遍历每一个房间，随机选择一个点
    room_num=D_x*D_y*D_z
    print('room_num:{}'.format(room_num))  # 房间的总数
    for room in range(int(room_num)): # 遍历所有房间
        temp=[]
        p_num=0  # 统计房间内点的数目
        # print('room:{}'.format(room))
        for i in range(point_cloud.shape[0]): # 遍历所有点
            if pt[i,-1]==room:  # 如果这个点在房间内,累加，并查询下一个点
                p_num+=1  # 房间内点的+1
                temp.append(pt[i,:-1])
        if p_num!=0: # 如果房间有点 才有接下来的计算
            temp_p=np.array(temp,dtype=np.float32)  # 应该是一个M*3的矩阵 , M为该房间点的个数

            # 采用随机选择分方式来选取降采样点
            choice = np.random.choice(len(temp_p), 1, replace=True) # 每个房间随机选择1个点,返回一个索引
            # print('choice:{}'.format(choice))  #
            p_random=temp_p[choice,:]  # 根据索引选择每一个房间的点
            p_random=p_random[0,:]   # 选择后地不是一个1*3的矩阵
            # print('p_random:{}'.format(p_random))
            filtered_points.append(p_random)  # 保存每一个房间的随机值

        if room*100/room_num%10==0:
            print('完成进度：{}%'.format(room*100/room_num))

    print('random filtered_points num :{}'.format(len(filtered_points))) # 降采样后点的数目
    # print('random filtered_points:{}'.format(filtered_points))
    # 把点云格式改成array，并对外返回
    filtered_points = np.array(filtered_points, dtype=np.float64)
    # print('filtered_points:{}'.format(filtered_points))
    return filtered_points


def main():
    # # 从ModelNet数据集文件夹中自动索引路径，加载点云
    # cat_index = 10 # 物体编号，范围是0-39，即对应数据集中40个物体
    # root_dir = '/Users/renqian/cloud_lesson/ModelNet40/ply_data_points' # 数据集路径
    # cat = os.listdir(root_dir)
    # filename = os.path.join(root_dir, cat[cat_index],'train', cat[cat_index]+'_0001.ply') # 默认使用第一个点云
    # point_cloud_pynt = PyntCloud.from_file(file_name)

    # # 加载自己的点云文件
    # file_name = "/Users/renqian/Downloads/program/cloud_data/11.ply"
    # point_cloud_pynt = PyntCloud.from_file(file_name)
    #
    # # 转成open3d能识别的格式
    # point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)
    # o3d.visualization.draw_geometries([point_cloud_o3d]) # 显示原始点云

    # 读取点云文件 txt
    file_name = 'E:\\深蓝学院\\Data\\modelnet40_normal_resampled\\airplane\\airplane_0001.txt'
    points=np.loadtxt(file_name,delimiter=',', dtype=np.float32)
    points=points[:,0:3]
    print('points shape:{}'.format(points.shape))
    print('total points number is:', points.shape[0])
    # 显示原始点云并返回一个open3d的点云类型
    point_cloud_o3d = visualize(points)


    # # 调用voxel滤波函数，实现滤波
    filtered_cloud = voxel_filter_random(points, 0.1)  # 调用函数
    # filtered_cloud = voxel_filter_mean(points, 0.1)  # 调用函数

    # print('filtered_cloud:{}'.format(filtered_cloud))
    point_cloud_o3d.points = o3d.utility.Vector3dVector(filtered_cloud)  # mat转为pointcloud类型
    # 显示滤波后的点云
    o3d.visualization.draw_geometries([point_cloud_o3d],width=800,height=600)

if __name__ == '__main__':
    main()
