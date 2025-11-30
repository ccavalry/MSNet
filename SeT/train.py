# 导入PyTorch核心库
import torch
# 从当前包的detect模块中导入detect函数（用于生成异常检测图）
from .detect import detect
# 从tqdm库导入tqdm类，用于显示训练进度条
from tqdm import tqdm
# 从metric模块导入roc_auc函数（用于计算ROC曲线和AUC分数）
from metric import roc_auc
# 导入numpy数值计算库，用于数组操作
import numpy as np



# 定义模型训练函数
# 输入：
#   x: 输入数据，PyTorch张量
#   model: 待训练的神经网络模型
#   criterion: 损失函数
#   cri_kwargs: 损失函数的额外参数，字典类型
#   epochs: 训练轮数
#   optimizer: 优化器
#   verbose: 是否显示训练进度
# 功能：执行模型的前向传播、损失计算、反向传播和参数更新
def train_model(
        x,              # 输入数据张量
        model,          # 待训练的模型
        criterion,      # 损失函数
        cri_kwargs,     # 损失函数的额外参数
        epochs,         # 训练轮数
        optimizer,      # 优化器
        verbose):       # 是否显示训练进度

    # 创建一个迭代器，用于遍历训练轮数
    epoch_iter = iter(_ for _ in range(epochs))
    # 如果需要显示进度，则使用tqdm包装迭代器，生成进度条
    if verbose:
        epoch_iter = tqdm(list(epoch_iter))
    # 遍历每一轮训练
    for _ in epoch_iter:

        # 清除优化器的梯度信息，避免梯度累积
        optimizer.zero_grad()

        # 前向传播：将输入x传入模型，得到输出y
        y = model(x)

        # 计算损失：调用损失函数，传入输入x、输出y和额外参数
        loss = criterion(x=x, y=y, **cri_kwargs)

        # 反向传播：计算损失对模型参数的梯度
        loss.backward()

        # 更新网络参数：根据梯度信息更新模型参数
        optimizer.step()

        # 如果需要显示进度，则更新进度条的后缀信息，显示当前损失值
        if verbose:
            epoch_iter.set_postfix({'loss': '{0:.4f}'.format(loss)})



# 定义分离训练主函数
# 输入：
#   x: 输入数据，PyTorch张量，形状为(rows, cols, bands)
#   gt: 真实异常标签，numpy数组，形状为(rows, cols)
#   model: 待训练的神经网络模型
#   loss: 损失函数
#   mask: 掩码对象，用于分离训练中的掩码操作
#   optimizer: 优化器
#   epochs: 每轮子迭代的训练轮数
#   output_iter: 输出检测图的子迭代次数
#   max_iter: 最大子迭代次数
#   verbose: 是否显示训练进度
# 输出：
#   output_dm: 最终的异常检测图，numpy数组，形状为(rows, cols)
#   history: AUC分数的历史记录，列表类型，长度为max_iter
def separation_training(
        x: torch.Tensor,      # 输入数据张量
        gt: np.ndarray,        # 真实异常标签
        model,                 # 待训练的模型
        loss,                  # 损失函数
        mask,                  # 掩码对象
        optimizer,             # 优化器
        epochs,                # 每轮子迭代的训练轮数
        output_iter,           # 输出检测图的子迭代次数
        max_iter,              # 最大子迭代次数
        verbose) -> (np.ndarray, list):  # 返回检测图和AUC历史
    """
    分离训练算法的主流程：
    1. 迭代训练模型
    2. 使用当前模型生成检测图
    3. 更新掩码
    4. 评估检测性能
    5. 记录AUC分数
    """

    # 初始化AUC分数历史记录列表
    history = []
    # 初始化输出检测图为与真实标签形状相同的零数组
    output_dm = np.zeros_like(gt)

    # 遍历子迭代次数（从1到max_iter）
    for i in range(1, max_iter + 1):
        # 如果需要显示进度，则打印当前子迭代次数
        if verbose:
            print('Iter {0}'.format(i))

        # 设置模型输入为原始输入x
        model_input = x

        # 训练模型：调用train_model函数，训练epochs轮
        train_model(
            model_input,      # 模型输入
            model,            # 待训练模型
            loss,             # 损失函数
            {'mask': mask},   # 损失函数的额外参数，包含掩码
            epochs,           # 训练轮数
            optimizer,        # 优化器
            verbose           # 是否显示进度
        )

        # 更新掩码：
        # 1. 使用当前模型生成检测图
        dm = detect(x, model(model_input))
        # 2. 使用检测图更新掩码
        mask.update(dm.detach())

        # 评估检测性能：
        # 1. 将检测图从PyTorch张量转换为numpy数组
        np_dm = dm.cpu().detach().numpy()
        # 2. 计算ROC曲线和AUC分数
        fpr, tpr, auc = roc_auc(np_dm, gt)
        # 3. 如果需要显示进度，则打印当前AUC分数
        if verbose:
            print('Current AUC score: {0:.4f}'.format(auc))

        # 记录AUC分数到历史记录列表
        history.append(auc)

        # 如果当前子迭代次数等于output_iter，则保存当前检测图作为最终输出
        if i == output_iter:
            output_dm = np_dm

    # 返回最终的检测图和AUC分数历史记录
    return output_dm, history
