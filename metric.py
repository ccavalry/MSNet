# 导入scikit-learn的metrics模块，用于计算ROC曲线和AUC分数
from sklearn import metrics
# 导入numpy数值计算库，用于数组操作
import numpy as np


# 定义ROC曲线和AUC分数计算函数
# 输入：
#   dm: 异常检测图，形状为(rows, cols)的numpy数组，值越大表示异常程度越高
#   gt: 真实异常标签，形状为(rows, cols)的numpy数组，值为0（背景）或1（异常）
# 输出：
#   fpr: 假阳性率数组，每个元素对应一个阈值下的假阳性率
#   tpr: 真阳性率数组，每个元素对应一个阈值下的真阳性率
#   auc: ROC曲线下的面积，即AUC分数，范围0-1，越大表示检测性能越好
def roc_auc(dm: np.ndarray,
            gt: np.ndarray):
    # 获取真实标签的空间维度：行数和列数
    rows, cols = gt.shape

    # 将2D真实标签重塑为1D数组，形状为(rows*cols,)
    gt = gt.reshape(rows * cols)
    # 将2D检测图重塑为1D数组，形状为(rows*cols,)
    dm = dm.reshape(rows * cols)

    # 计算ROC曲线：
    # 输入为真实标签gt和检测图dm
    # 输出为假阳性率(fpr)、真阳性率(tpr)和对应的阈值
    # _表示忽略阈值输出
    fpr, tpr, _ = metrics.roc_curve(gt, dm)
    # 计算ROC曲线下的面积，即AUC分数
    auc = metrics.auc(fpr, tpr)

    # 返回ROC曲线的假阳性率、真阳性率和AUC分数
    return fpr, tpr, auc



