# 导入scipy的io模块，用于读取MAT格式的数据集文件
import scipy.io as sio
# 导入操作系统接口模块，用于文件路径操作
import os


# 定义数据集名称为'coast'
name = 'coast'
# 获取当前Python文件所在的目录路径
path = os.path.dirname(__file__)
# 构建MAT文件的完整路径：当前目录下的'coast.mat'文件
file_name = os.path.join(path, name + '.mat')


# 定义获取数据集的函数
# 输出：
#   data: 高光谱数据，形状为(rows, cols, bands)的numpy数组，数据类型为float
#   gt: 真实异常标签，形状为(rows, cols)的numpy数组，数据类型为bool（True表示异常，False表示背景）
def get_data():
    # 读取MAT文件，返回一个字典，键为MAT文件中的变量名，值为对应的数据
    mat = sio.loadmat(file_name)
    # 从MAT文件中获取高光谱数据，键为'data'，并转换为float类型
    data = mat['data'].astype(float)
    # 从MAT文件中获取真实异常标签，键为'map'，并转换为bool类型
    gt = mat['map'].astype(bool)

    # 返回高光谱数据和真实异常标签
    return data, gt
