# 导入PyTorch神经网络模块
from torch import nn
# 导入PyTorch函数模块，包含卷积、池化等函数
import torch.nn.functional as F
# 导入PyTorch核心库
import torch


# 定义拉普拉斯高斯（Laplacian of Gaussian，LoG）算子类
# 用于计算图像的边缘信息，增强异常区域的特征
class LoG(nn.Module):

    # 初始化方法
    # 参数：
    #   device: 计算设备（CPU或GPU）
    def __init__(self, device):
        # 调用父类nn.Module的初始化方法
        super(LoG, self).__init__()

        # 定义5x5的LoG卷积核模板
        self.tmpl = torch.tensor(
            [[-2, -4, -4, -4, -2],
             [-4,  0,  8,  0, -4],
             [-4,  8, 24,  8, -4],
             [-4,  0,  8,  0, -4],
             [-2, -4, -4, -4, -2]]
        ).to(device).float()  # 将模板移至指定设备并转换为float类型

        # 获取模板的空间维度（宽度和高度，这里是5x5）
        ws, ws = self.tmpl.shape
        # 将2D模板重塑为3D卷积核，形状为(1, 1, 1, ws, ws)
        # 维度顺序：(out_channels, in_channels, depth, height, width)
        self.tmpl = self.tmpl.reshape(1, 1, 1, ws, ws)
        # 计算填充大小：模板宽度的一半，用于保持卷积后空间维度不变
        self.pad = ws // 2

    # 前向传播方法
    # 参数：
    #   x: 输入张量，形状为(rows, cols, bands)
    # 返回：
    #   应用LoG算子后的张量，形状为(rows, cols, bands)
    def forward(self, x):
        # 调整输入张量的形状：
        # permute(2, 0, 1)将形状从(rows, cols, bands)转换为(bands, rows, cols)
        # unsqueeze(0)添加批次维度，形状变为(1, bands, rows, cols)
        x = x.permute(2, 0, 1).unsqueeze(0)

        # 应用反射填充：在输入的四周添加pad个像素，使用反射模式
        # 填充顺序：(left, right, top, bottom)
        x = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode='reflect')
        # 添加深度维度，形状变为(1, 1, bands, rows+2*pad, cols+2*pad)
        x = x.unsqueeze(0)

        # 对每个波段应用3D卷积，使用预定义的LoG模板
        # 3D卷积会对深度（波段）维度进行卷积
        x = F.conv3d(x, self.tmpl)

        # 调整输出张量的形状：
        # squeeze(0).squeeze(0)移除前两个维度，形状变为(bands, rows, cols)
        x = x.squeeze(0).squeeze(0)
        # permute(1, 2, 0)将形状转换回(rows, cols, bands)
        x = x.permute(1, 2, 0)

        # 返回应用LoG算子后的张量
        return x


# 定义分离训练损失（Separation Training Loss，SeTLoss）类
class SeTLoss(nn.Module):

    # 初始化方法
    # 参数：
    #   lmda: 异常抑制损失的权重系数
    #   device: 计算设备（CPU或GPU）
    def __init__(self, lmda, device):
        # 调用父类nn.Module的初始化方法
        super(SeTLoss, self).__init__()

        # 保存异常抑制损失的权重系数
        self.lmda = lmda
        # 定义一个极小值，用于避免除零错误
        self.eps = 1e-6

        # 初始化LoG算子实例
        self.log = LoG(device)
        # 定义L2范数计算函数：计算张量元素的平方和
        self.norm = lambda x: (x ** 2).sum()

    # 前向传播方法
    # 参数：
    #   **kwargs: 关键字参数，包含mask（掩码）、x（输入数据）、y（解码器输出）
    # 返回：
    #   分离训练损失值，标量
    def forward(self, **kwargs):
        # 从关键字参数中获取掩码
        anm_mask = kwargs['mask']
        # 从关键字参数中获取输入数据x
        x = kwargs['x']
        # 从关键字参数中获取解码器输出y
        decoder_outputs = kwargs['y']
        # 获取第一个解码器输出（主输出）
        y = decoder_outputs[0]

        # 计算异常区域的像素数量
        num_anm = anm_mask.count()

        # 获取背景掩码：异常掩码的非操作
        bg_mask = anm_mask.not_op()
        # 计算背景区域的像素数量
        num_bg = bg_mask.count()

        # 对重建图像y应用LoG算子，增强边缘信息
        log_y = self.log(y)

        # 计算异常抑制损失：
        # 1. anm_mask.dot_prod(log_y)：异常掩码与LoG结果的逐元素乘积
        # 2. self.norm(...)：计算L2范数
        # 3. 除以异常像素数量+eps，避免除零
        as_loss = self.norm(anm_mask.dot_prod(log_y)) / (num_anm + self.eps)

        # 计算背景重建损失：
        # 1. bg_mask.dot_prod(x - y)：背景掩码与重建误差的逐元素乘积
        # 2. self.norm(...)：计算L2范数
        # 3. 除以背景像素数量
        br_loss = self.norm(bg_mask.dot_prod(x - y)) / num_bg

        # 计算分离训练总损失：背景重建损失 + 权重系数*异常抑制损失
        set_loss = br_loss + self.lmda * as_loss

        # 返回分离训练损失值
        return set_loss


# 定义多尺度重建损失（Multi-Scale Reconstruction Loss，MSRLoss）类
class MSRLoss(nn.Module):

    # 初始化方法
    def __init__(self):
        # 调用父类nn.Module的初始化方法
        super(MSRLoss, self).__init__()

        # 定义L2范数计算函数：计算张量元素的平方和
        self.norm = lambda x: (x ** 2).sum()

    # 前向传播方法
    # 参数：
    #   **kwargs: 关键字参数，包含x（输入数据）、y（解码器输出）
    # 返回：
    #   多尺度重建损失值，标量
    def forward(self, **kwargs):
        # 从关键字参数中获取输入数据x
        x = kwargs['x']
        # 从关键字参数中获取解码器输出列表
        decoder_outputs = kwargs['y']

        # 初始化缩放因子为1
        scale = 1
        # 初始化损失列表，用于存储各尺度的损失
        layers = []

        # 遍历每个解码器输出
        for _do in decoder_outputs:
            # 计算当前尺度下的空间维度
            _rows = _do.shape[0] // scale
            _cols = _do.shape[1] // scale
            # 对输入数据x进行下采样：
            # permute(2, 0, 1)将形状从(rows, cols, bands)转换为(bands, rows, cols)
            # unsqueeze(0)添加批次维度，形状变为(1, bands, rows, cols)
            # 使用双线性插值下采样到(_rows, _cols)
            _x_down = F.interpolate(
                x.permute(2, 0, 1).unsqueeze(0),
                size=(_rows, _cols), mode='bilinear'
            )
            # 对解码器输出_do进行下采样，处理方式与x相同
            _do_down = F.interpolate(
                _do.permute(2, 0, 1).unsqueeze(0),
                size=(_rows, _cols), mode='bilinear'
            )
            # 计算当前尺度下的重建损失：
            # 1. 计算下采样后x与_do的差异
            # 2. 计算差异的L2范数
            # 3. 除以当前尺度下的像素数量，归一化损失
            _layer = self.norm(_x_down - _do_down) / (_rows * _cols)
            # 将当前尺度的损失添加到损失列表
            layers.append(_layer)
            # 缩放因子乘以2，用于下一个尺度
            scale *= 2

        # 计算多尺度重建损失：所有尺度损失的平均值
        msr_loss = sum(layers) / len(layers)
        # 返回多尺度重建损失值
        return msr_loss


# 定义总损失（TotalLoss）类，结合分离训练损失和多尺度重建损失
class TotalLoss(nn.Module):

    # 初始化方法
    # 参数：
    #   lmda: 异常抑制损失的权重系数
    #   device: 计算设备（CPU或GPU）
    def __init__(self, lmda, device):
        # 调用父类nn.Module的初始化方法
        super(TotalLoss, self).__init__()

        # 初始化分离训练损失
        self.set_loss = SeTLoss(lmda, device)
        # 初始化多尺度重建损失
        self.msr_loss = MSRLoss()

    # 前向传播方法
    # 参数：
    #   **kwargs: 关键字参数，包含x（输入数据）、y（解码器输出）、mask（掩码）
    # 返回：
    #   总损失值，标量
    def forward(self, **kwargs):
        # 获取输入数据x的形状：行数、列数、波段数
        rows, cols, bands = kwargs['x'].shape
        # 计算总损失：分离训练损失 + 多尺度重建损失
        total_loss = self.set_loss(**kwargs) + self.msr_loss(**kwargs)
        # 将总损失除以波段数，进行归一化
        total_loss /= bands
        # 返回总损失值
        return total_loss
