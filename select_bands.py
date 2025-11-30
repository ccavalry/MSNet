# 导入numpy数值计算库，用于矩阵运算和数组操作
import numpy as np


# 定义OPBS（Orthogonal Projection-based Band Selection）波段选择函数
# 输入：
#   x: 高光谱数据，形状为(rows, cols, bands)的numpy数组
#   num_bs: 要选择的波段数量
# 输出：
#   选中的波段索引列表，按升序排列
def OPBS(
        x: np.ndarray,
        num_bs: int
):
    """
    参考文献：
    W. Zhang, X. Li, Y. Dou, and L. Zhao, "A geometry-based band
    selection approach for hyperspectral image analysis," IEEE Transactions
    on Geoscience and Remote Sensing, vol. 56, no. 8, pp. 4318–4333, 2018.
    """
    # 获取输入高光谱数据的维度：行数、列数、波段数
    rows, cols, bands = x.shape
    # 定义一个极小值，用于避免除零错误
    eps = 1e-9

    # 将3D高光谱数据重塑为2D数组，形状为(rows*cols, bands)，每一行代表一个像素
    x_2d = np.reshape(x, (rows * cols, bands))
    # 创建数据副本，用于后续的正交投影计算
    y_2d = x_2d.copy()
    # 初始化能量数组，用于存储每个波段的能量值
    h = np.zeros(bands)
    # 初始化选中波段索引列表
    band_idx = []

    # 第一步：选择方差最大的波段作为第一个选中波段
    # 计算每个波段的方差，选择方差最大的波段索引
    idx = np.argmax(np.var(x_2d, axis=0))
    # 将选中的波段索引添加到列表中
    band_idx.append(idx)
    # 计算选中波段的能量（平方和）
    h[idx] = np.sum(x_2d[:, band_idx[-1]] ** 2)

    # 初始化计数器，已经选中了一个波段
    i = 1
    # 循环直到选中了num_bs个波段
    while i < num_bs:
        # 获取上一次选中的波段索引
        id_i_1 = band_idx[i - 1]

        # 初始化最大能量值和对应的波段索引
        _elem, _idx = -np.inf, 0
        # 遍历所有波段
        for t in range(bands):
            # 如果该波段还没有被选中
            if t not in band_idx:
                # 计算当前波段与上一个选中波段的正交分量
                # 公式：y_2d[:, t] = y_2d[:, t] - proj(y_2d[:, t], y_2d[:, id_i_1])
                # 其中proj(a, b) = b * (a·b / ||b||²)
                y_2d[:, t] = y_2d[:, t] - y_2d[:, id_i_1] * (np.dot(y_2d[:, id_i_1], y_2d[:, t]) / (h[id_i_1] + eps))
                # 计算当前波段正交分量的能量（平方和）
                h[t] = np.dot(y_2d[:, t], y_2d[:, t])

                # 如果当前波段的能量大于最大能量值
                if h[t] > _elem:
                    # 更新最大能量值
                    _elem = h[t]
                    # 更新对应的波段索引
                    _idx = t

        # 将当前选中的波段索引添加到列表中
        band_idx.append(_idx)
        # 计数器加1
        i += 1

    # 对选中的波段索引进行排序，返回升序排列的波段索引列表
    band_idx = sorted(band_idx)
    return band_idx
