# 实现PCA分析和法向量计算，并加载数据集中的文件进行验证

import open3d as o3d # 可视化
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

# 功能：计算PCA的函数
# 输入：
#     data：点云，NX3的矩阵
#     correlation：区分np的cov和corrcoef，不输入时默认为False
#     sort: 特征值排序，排序是为了其他功能方便使用，不输入时默认为True
# 输出：
#     eigenvalues：特征值
#     eigenvectors：特征向量
def PCA(data, correlation=False, sort=True):
    # 作业1
    # 屏蔽开始
    # 1.去中心化
    # 2.计算协方差矩阵得到特征值，特征向量

    # 1.去中心化
    pt_mean=np.mean(data,axis=0) # 1x3
    data=data-pt_mean # Nx3

    # 2.计算协方差矩阵
    data_T=data.T
    C=np.matmul(data_T,data) # 3x3
    eigenvalues,eigenvectors=np.linalg.eig(C)  # 特征值，特征向量

    # 屏蔽结束

    if sort:
        sort = eigenvalues.argsort()[::-1]  # sort为索引,由大到小排列
        eigenvalues = eigenvalues[sort]  # 由大到小
        eigenvectors = eigenvectors[:, sort]  # 由大到小

    return eigenvalues, eigenvectors


def main():
    '''
    # 指定点云路径
    # cat_index = 10 # 物体编号，范围是0-39，即对应数据集中40个物体
    # root_dir = '/Users/renqian/cloud_lesson/ModelNet40/ply_data_points' # 数据集路径
    # cat = os.listdir(root_dir)
    # filename = os.path.join(root_dir, cat[cat_index],'train', cat[cat_index]+'_0001.ply') # 默认使用第一个点云

    # 加载原始点云
    point_cloud_pynt = PyntCloud.from_file("/Users/renqian/Downloads/program/cloud_data/11.ply")
    point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)
    # o3d.visualization.draw_geometries([point_cloud_o3d]) # 显示原始点云

    # 从点云中获取点，只对点进行处理
    points = point_cloud_pynt.points
    '''

    # 直接读取txt文件,逗号隔开
    points=np.loadtxt('E:\\深蓝学院\\Data\\modelnet40_normal_resampled\\airplane\\airplane_0001.txt',\
                      delimiter=',', dtype=np.float32)
    points=points[:,0:3]
    print('point shape:{}'.format(points.shape))
    print('total points number is:', points.shape[0])

    # 显示原始点云
    point_cloud_o3d=visualize(points)

    # 用PCA分析点云主方向
    w, v = PCA(points)
    point_cloud_vector = v[:, 2] # 点云主方向对应的向量 - 最小特征值对应的向量
    print('the main orientation of this pointcloud is: ', point_cloud_vector)
    # TODO: 此处只显示了点云，还没有显示PCA
    # o3d.visualization.draw_geometries([point_cloud_o3d])
    
    # 循环计算每个点的法向量
    pcd_tree = o3d.geometry.KDTreeFlann(point_cloud_o3d)
    normals = []
    # 作业2
    # 屏蔽开始

    # print('point_cloud_o3d.points:{}'.format(point_cloud_o3d.points)) # 数据类型
    # 遍历每一个点，选取最近的20个点
    K=19  # 选取的近邻点数目
    for i in range(points.shape[0]):
        [k,index,_]=pcd_tree.search_knn_vector_3d(point_cloud_o3d.points[i],K+1)

        eigenvalues,eigenvectors=PCA(points[index[:],:])
        # print('index[1]:{},points[index[1],:]:{}'.format(index[1],points[index[1],:]))

        normals.append(eigenvectors[:,-1])   # 最小特征值对应的特征向量为法向量
    # 由于最近邻搜索是第二章的内容，所以此处允许直接调用open3d中的函数

    # 屏蔽结束
    normals = np.array(normals, dtype=np.float64)
    # TODO: 此处把法向量存放在了normals中
    point_cloud_o3d.normals = o3d.utility.Vector3dVector(normals)
    o3d.visualization.draw_geometries([point_cloud_o3d],width=800,height=600)


if __name__ == '__main__':
    main()
