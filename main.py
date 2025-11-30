# 从dataset模块导入coast数据集相关功能/类
from dataset import coast
# 从model模块导入MSNet模型类
from model import MSNet
# 导入matplotlib的pyplot模块，用于图像绘制与保存
import matplotlib.pyplot as plt
# 从PyTorch优化器模块导入Adam优化器
from torch.optim import Adam
# 导入波段选择相关模块（包含波段选择算法）
import select_bands
# 导入PyTorch深度学习框架核心库
import torch
# 导入工具类模块（包含数据归一化等工具函数）
import utils
# 导入指标计算模块（用于评估模型性能）
import metric
# 导入操作系统接口模块，用于文件路径操作
import os
# 从SeT模块导入自定义的损失函数、掩码类和训练函数
from SeT import (
    TotalLoss,  # 总损失函数类（包含任务损失与正则项）
    Mask,  # 掩码生成类（用于分离训练中的掩码操作）
    separation_training  # 分离训练主函数
)

# ======================== 参数设置部分 ========================
# 设置计算设备：优先使用CUDA（GPU），无GPU则使用CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lmda = 1e-3  # 损失函数中的正则化系数λ
num_bs = 64  # 待选择的波段数量
num_layers = 3  # MSNet模型的网络层数
lr = 1e-3  # 优化器的学习率
epochs = 150  # 训练总轮数
output_iter = 5  # 训练过程中信息打印的间隔迭代数
max_iter = 10  # 分离训练中的最大迭代次数（子迭代）
data_norm = True  # 是否对输入数据进行归一化处理
Net = MSNet  # 指定使用的模型架构为MSNet
net_kwargs = dict()  # 初始化模型参数字典（用于传递给模型构造函数）
net_kwargs['num_layers'] = num_layers  # 向模型参数字典添加层数参数

# ======================== 数据加载部分 ========================
dataset = coast  # 实例化coast数据集对象
data, gt = dataset.get_data()  # 获取数据集的原始高光谱数据和异常检测真实标签（ground truth）
rows, cols, bands = data.shape  # 获取数据维度：行数、列数、波段数
net_kwargs['shape'] = (rows, cols, num_bs)  # 向模型参数添加输入数据的空间维度与波段数
print('Detecting on %s...' % dataset.name)  # 打印当前检测的数据集名称

# ======================== 数据预处理部分 ========================
band_idx = select_bands.OPBS(data, num_bs)  # 使用OPBS算法选择num_bs个最优波段，返回波段索引
data_bs = data[:, :, band_idx]  # 根据选中的波段索引提取对应波段数据
if data_norm:  # 如果开启数据归一化
    # 使用Z-Score归一化（均值为0，标准差为1）：先拟合数据分布，再转换数据
    data_bs = utils.ZScoreNorm().fit(data_bs).transform(data_bs)

# ======================== 模型初始化部分 ========================
# 实例化MSNet模型，传入参数并移至指定计算设备，数据类型设为float32
model = Net(**net_kwargs).to(device).float()

# ======================== 损失与工具类初始化 ========================
loss = TotalLoss(lmda, device)  # 实例化总损失函数，传入正则系数和计算设备
mask = Mask((rows, cols), device)  # 实例化掩码类，传入数据空间维度和计算设备

# ======================== 优化器初始化 ========================
# 初始化Adam优化器，传入模型可训练参数和学习率
optimizer = Adam(model.parameters(), lr=lr)

# ======================== 分离训练执行 ========================
# 将预处理后的numpy数组转换为PyTorch张量，移至指定设备并转为float32
x_bs = torch.from_numpy(data_bs).to(device).float()
# 执行分离训练，返回检测图（pr_dm）和训练历史（history）
pr_dm, history = separation_training(
    x=x_bs,  # 输入的高光谱数据张量
    gt=gt,  # 真实异常标签
    model=model,  # 待训练的模型
    loss=loss,  # 损失函数
    mask=mask,  # 掩码工具
    optimizer=optimizer,  # 优化器
    epochs=epochs,  # 训练轮数
    output_iter=output_iter,  # 信息打印间隔
    max_iter=max_iter,  # 子迭代最大次数
    verbose=True  # 是否打印训练过程详细信息
)

# ======================== 结果保存与可视化 ========================
# 构建结果保存路径：results文件夹下的模型名称子文件夹
result_path = os.path.join('results', model.name)
if not os.path.exists(result_path):  # 如果路径不存在则创建
    os.makedirs(result_path)

# 计算传统RX算法的检测结果，用于对比
rx_dm = utils.rx(data)
# 计算RX算法的ROC曲线和AUC分数
fpr, tpr, rx_auc = metric.roc_auc(rx_dm, gt)
# 绘制RX算法的ROC曲线，添加标签
plt.plot(fpr, tpr, label='RX: %.4f' % rx_auc)

# 计算MSNet+SeT算法的ROC曲线和AUC分数
fpr, tpr, pr_auc = metric.roc_auc(pr_dm, gt)
# 绘制MSNet+SeT算法的ROC曲线，添加标签，使用黑色实线
plt.plot(fpr, tpr, label='%s+SeT: %.4f' % (model.name, pr_auc),
         c='black', alpha=0.7)

# 添加网格线，透明度为0.3
plt.grid(alpha=0.3)
# 添加图例
plt.legend()
# 保存ROC曲线为PDF文件
plt.savefig(
    os.path.join(result_path, '%s_roc.pdf' % dataset.name)
)
# 清空当前图形
plt.clf()
# 关闭当前图形窗口
plt.close()

# 生成迭代次数列表：每个子迭代结束时的总训练轮数
iters = [(_ + 1) * epochs for _ in range(max_iter)]
# 设置X轴刻度为迭代次数列表
plt.xticks(iters)
# 绘制AUC分数随迭代次数的变化曲线
plt.plot(iters, history)
# 在最佳迭代点（output_iter）添加标记点，用于指示最佳停止位置
plt.scatter([output_iter * epochs], [history[output_iter - 1]],
            marker='o', edgecolors='black', facecolors='white', label='Stop',
            zorder=10)  # zorder=10确保标记点显示在最上层
# 添加网格线，透明度为0.3
plt.grid(alpha=0.3)
# 设置X轴标签为"Epoch"
plt.xlabel('Epoch')
# 设置Y轴标签为"AUC"
plt.ylabel('AUC')
# 添加图例
plt.legend()
# 保存AUC历史曲线为PDF文件
plt.savefig(
    os.path.join(result_path, '%s_auc_history.pdf' % dataset.name)
)
# 清空当前图形
plt.clf()
# 关闭当前图形窗口
plt.close()

# 打印完成提示
print('Complete.')
# 打印结果保存路径
print('Results are saved in results/.')


