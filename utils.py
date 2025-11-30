# 导入numpy数值计算库，用于矩阵运算和数组操作
import numpy as np


# 定义RX异常检测算法函数
# 输入：
#   x: 高光谱数据，形状为(rows, cols, bands)的numpy数组
# 输出：
#   dm: 异常检测图，形状为(rows, cols)的numpy数组，值越大表示异常程度越高
def rx(x: np.ndarray):
    # 获取输入数据的维度：行数、列数、波段数
    rows, cols, bands = x.shape
    # 将3D高光谱数据重塑为2D数组，形状为(rows*cols, bands)，每一行代表一个像素
    x_2d = x.reshape((rows * cols, bands))

    # 计算协方差矩阵的逆矩阵：
    # 1. 先计算数据的协方差矩阵（x_2d.T表示转置，每一列代表一个波段）
    # 2. 然后计算协方差矩阵的逆矩阵
    inv_Sig = np.linalg.inv(np.cov(x_2d.T))
    # 计算每个波段的均值向量，形状为(1, bands)
    mu = np.mean(x_2d, axis=0, keepdims=True)
    # 定义马氏距离计算函数：
    # 马氏距离公式：(_x - mu) @ inv_Sig @ (_x - mu).T
    # 其中@表示矩阵乘法，.T表示转置
    get_M_dis = lambda _x: (_x - mu) @ inv_Sig @ (_x - mu).T
    # 对每个像素计算马氏距离，生成异常检测图
    dm = np.array([get_M_dis(_x) for _x in x_2d])

    # 将2D检测图重塑为3D输入数据的空间维度，形状为(rows, cols)
    dm = dm.reshape((rows, cols))

    # 返回异常检测图
    return dm


# 定义最小-最大归一化类（Min-Max Normalization）
class MinMaxNorm:

    # 初始化函数
    # 参数：
    #   feature_range: 归一化后的取值范围，默认为(0, 1)
    def __init__(self, feature_range=(0, 1)):
        # 保存归一化范围
        self.feature_range = feature_range
        # 初始化最小值和最大值为None，将在fit方法中计算
        self.min = None
        self.max = None

    # 拟合数据分布，计算最小值和最大值
    # 参数：
    #   x: 输入数据，用于计算最小值和最大值
    # 返回：
    #   self: 类实例本身，支持链式调用
    def fit(self, x):
        # 计算数据的最小值
        self.min = x.min()
        # 计算数据的最大值
        self.max = x.max()
        # 返回实例本身，支持链式调用（如MinMaxNorm().fit(x).transform(x)）
        return self

    # 对数据进行归一化转换
    # 参数：
    #   x: 待归一化的数据
    # 返回：
    #   x_norm: 归一化后的数据
    def transform(self, x):
        # 计算标准化数据：(x - min) / (max - min)，将数据缩放到[0, 1]范围
        x_std = (x - self.min) / (self.max - self.min)
        # 将标准化数据缩放到指定的feature_range范围
        x_norm = x_std * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]
        # 返回归一化后的数据
        return x_norm

    # 对归一化数据进行逆转换，恢复原始数据范围
    # 参数：
    #   x_norm: 归一化后的数据
    # 返回：
    #   x: 恢复后的原始范围数据
    def inverse_transform(self, x_norm):
        # 将归一化数据转换回[0, 1]范围
        x_std = (x_norm - self.feature_range[0]) / (self.feature_range[1] - self.feature_range[0])
        # 将[0, 1]范围数据转换回原始数据范围
        x = x_std * (self.max - self.min) + self.min
        # 返回恢复后的原始数据
        return x


# 定义Z-Score归一化类（均值为0，标准差为1）
class ZScoreNorm:

    # 初始化函数
    def __init__(self):
        # 初始化均值和标准差为None，将在fit方法中计算
        self.means = None
        self.stds = None

    # 拟合数据分布，计算每个波段的均值和标准差
    # 参数：
    #   x: 输入数据，形状为(rows, cols, bands)的高光谱数据
    # 返回：
    #   self: 类实例本身，支持链式调用
    def fit(self, x):
        # 计算每个波段的均值：axis=(0, 1)表示对空间维度（行和列）求平均
        self.means = np.mean(x, axis=(0, 1))
        # 计算每个波段的标准差：axis=(0, 1)表示对空间维度求标准差
        self.stds = np.std(x, axis=(0, 1))
        # 返回实例本身，支持链式调用
        return self

    # 对数据进行Z-Score归一化转换
    # 参数：
    #   x: 待归一化的数据，形状为(rows, cols, bands)
    # 返回：
    #   x_norm: 归一化后的数据，每个波段的均值为0，标准差为1
    def transform(self, x):
        # 创建一个与输入数据形状相同的零数组，用于存储归一化后的数据
        x_norm = np.zeros_like(x)
        # 遍历每个波段
        for i in range(x.shape[2]):
            # 对每个波段进行Z-Score归一化：(x - mean) / std
            x_norm[:, :, i] = (x[:, :, i] - self.means[i]) / self.stds[i]
        # 返回归一化后的数据
        return x_norm

    # 对Z-Score归一化数据进行逆转换，恢复原始数据范围
    # 参数：
    #   x_norm: Z-Score归一化后的数据
    # 返回：
    #   x: 恢复后的原始数据
    def inverse_transform(self, x_norm):
        # 创建一个与输入数据形状相同的零数组，用于存储恢复后的数据
        x = np.zeros_like(x_norm)
        # 遍历每个波段
        for i in range(x_norm.shape[2]):
            # 对每个波段进行逆转换：x_norm * std + mean
            x[:, :, i] = x_norm[:, :, i] * self.stds[i] + self.means[i]
        # 返回恢复后的原始数据
        return x
