# 导入PyTorch核心库
import torch
# 从typing模块导入Union类型，用于类型注解（表示参数可以是多种类型之一）
from typing import Union
# 导入工具类模块（包含MinMaxNorm等工具函数）
import utils


# 定义掩码类，用于分离训练中的掩码操作
class Mask:

    # 初始化方法
    # 参数：
    #   init: 初始掩码或掩码形状，可以是tuple（表示形状）或torch.Tensor（表示初始掩码）
    #   device: 计算设备（CPU或GPU）
    def __init__(self,
                 init: Union[tuple, torch.Tensor],  # 初始掩码或掩码形状
                 device):  # 计算设备

        # 如果init是tuple类型（表示掩码形状）
        if isinstance(init, tuple):
            # 创建与形状init相同的零张量作为初始掩码，并移至指定设备
            self.mask = torch.zeros(init).to(device)
            # 在最后一个维度添加一个维度，形状变为(rows, cols, 1)
            self.mask = self.mask.unsqueeze(-1)
        # 如果init是torch.Tensor类型（表示初始掩码）
        elif isinstance(init, torch.Tensor):
            # 直接使用init作为掩码
            self.mask = init

    # 定义点积方法：计算掩码与输入张量的逐元素乘积
    # 参数：
    #   x: 输入张量，形状应与掩码兼容
    # 返回：
    #   掩码与x的逐元素乘积，形状与x相同
    def dot_prod(self, x):
        # 返回掩码与x的逐元素乘积
        return self.mask * x

    # 定义count方法：计算掩码中元素的总和
    # 返回：
    #   掩码元素的总和，标量
    def count(self):
        # 返回掩码元素的总和
        return self.mask.sum()

    # 定义not_op方法：生成掩码的非操作（1 - 掩码）
    # 返回：
    #   新的Mask对象，掩码为原掩码的非操作结果
    def not_op(self):
        # 创建新的Mask对象，掩码为1 - 原掩码，设备与原掩码相同
        return Mask(1 - self.mask, self.mask.device)

    # 定义update方法：使用检测图更新掩码
    # 参数：
    #   dm: 异常检测图，形状为(rows, cols)的torch.Tensor
    def update(self,
               dm: torch.Tensor):  # 异常检测图
        # 使用MinMaxNorm归一化检测图：
        # 1. 创建MinMaxNorm对象
        # 2. 拟合检测图的分布
        # 3. 将检测图转换为[0, 1]范围
        # 4. 在最后一个维度添加一个维度，形状变为(rows, cols, 1)
        self.mask = utils.MinMaxNorm().fit(dm).transform(dm).unsqueeze(-1)




