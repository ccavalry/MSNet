# 导入PyTorch神经网络模块
from torch import nn
# 从PyTorch函数模块导入插值函数，用于上采样
from torch.nn.functional import interpolate
# 导入PyTorch核心库
import torch


# 定义瓶颈模块（Bottleneck）：包含两个卷积层和残差连接
class Bottleneck(nn.Module):

    def __init__(self,
                 in_channels,      # 输入通道数
                 hidden_channels):  # 隐藏层通道数
        # 调用父类nn.Module的初始化方法
        super(Bottleneck, self).__init__()

        # 定义瓶颈模块的网络层序列：
        # 1. 3x3卷积，输入通道数->隐藏通道数，步长1，填充1（保持空间维度不变）
        # 2. ReLU激活函数
        # 3. 3x3卷积，隐藏通道数->输入通道数，步长1，填充1（保持空间维度不变）
        self.layers = nn.Sequential(*[
            nn.Conv2d(in_channels, hidden_channels, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, in_channels, 3, 1, 1),
        ])

    # 前向传播函数
    def forward(self, x):
        # 残差连接：将输入x与网络层输出相加
        x = self.layers(x) + x
        # 返回输出
        return x


# 定义下采样模块（Down）：用于降低特征图的空间分辨率
class Down(nn.Module):

    def __init__(self,
                 down_rate,):  # 下采样率，即池化核大小
        # 调用父类nn.Module的初始化方法
        super(Down, self).__init__()

        # 定义下采样模块的网络层序列：
        # 1. ReLU激活函数
        # 2. 平均池化，池化核大小为down_rate，步长默认等于池化核大小
        self.layers = nn.Sequential(*[
            nn.ReLU(),
            nn.AvgPool2d(down_rate)
        ])

    # 前向传播函数
    def forward(self, x):
        # 将输入x通过下采样模块
        return self.layers(x)


# 定义上采样模块（Up）：用于提高特征图的空间分辨率
class Up(nn.Module):

    def __init__(self,
                 shape,        # 目标输出形状，格式为(rows, cols, bands)
                 need_relu):   # 是否需要ReLU激活函数
        # 调用父类nn.Module的初始化方法
        super(Up, self).__init__()

        # 保存是否需要ReLU激活的标志
        self.need_relu = need_relu
        # 解析目标形状：行数、列数、通道数
        self.rows, self.cols, bands = shape

        # 定义1x1卷积层，用于调整通道数（这里输入输出通道数相同）
        self.conv = nn.Conv2d(bands, bands, 1)
        # 定义ReLU激活函数
        self.relu = nn.ReLU()

    # 前向传播函数
    def forward(self, x):
        # 双线性插值上采样：将输入x上采样到目标大小(rows, cols)
        x = interpolate(x, size=(self.rows, self.cols), mode='bilinear')
        # 通过1x1卷积层
        x = self.conv(x)
        # 如果需要ReLU激活，则应用ReLU
        if self.need_relu:
            x = self.relu(x)
        # 返回上采样后的特征图
        return x


# 定义编码器模块（Encoder）：包含瓶颈、下采样和上采样
class Encoder(nn.Module):

    def __init__(self,
                 shape,        # 输入形状，格式为(rows, cols, bands)
                 down_rate):   # 下采样率
        # 调用父类nn.Module的初始化方法
        super(Encoder, self).__init__()

        # 解析输入形状：行数、列数、通道数
        rows, cols, bands = shape

        # 初始化瓶颈模块：输入通道数bands，隐藏通道数16
        self.bottleneck = Bottleneck(bands, 16)
        # 初始化下采样模块：下采样率为down_rate
        self.down_block = Down(down_rate)
        # 初始化上采样模块：目标形状为输入形状，不需要ReLU激活
        self.up_block = Up(shape, need_relu=False)

    # 前向传播函数
    def forward(self, x):
        # 1. 通过瓶颈模块提取特征
        output = self.bottleneck(x)
        # 2. 通过下采样模块降低空间分辨率
        output = self.down_block(output)
        # 3. 通过上采样模块恢复空间分辨率，生成skip connection（跳连）特征
        sm = self.up_block(output)
        # 返回下采样后的特征和跳连特征
        return output, sm


# 定义解码器模块（Decoder）：包含上采样和卷积
class Decoder(nn.Module):

    def __init__(self,
                 shape):  # 输入形状，格式为(rows, cols, bands)
        # 调用父类nn.Module的初始化方法
        super(Decoder, self).__init__()

        # 解析输入形状：行数、列数、通道数
        rows, cols, bands = shape

        # 初始化上采样模块：目标形状为输入形状，需要ReLU激活
        self.up_block = Up(shape, need_relu=True)
        # 初始化1x1卷积层，用于调整通道数（这里输入输出通道数相同）
        self.conv = nn.Conv2d(bands, bands, 1)

    # 前向传播函数
    def forward(self, encoder_output, sm):
        # 1. 将编码器输出通过上采样模块恢复空间分辨率
        output = self.up_block(encoder_output)
        # 2. 与跳连特征sm相加
        output = output + sm
        # 3. 通过1x1卷积层调整特征
        output = self.conv(output)
        # 返回解码后的特征
        return output


# 定义MSNet主网络类：多尺度网络，包含多个编码器和解码器
class MSNet(nn.Module):

    def __init__(self,
                 **kwargs,):  # 可变参数，包含网络层数和输入形状等
        # 调用父类nn.Module的初始化方法
        super(MSNet, self).__init__()

        # 设置网络名称
        self.name = 'MSNet'

        # 从参数中获取网络层数
        self.num_layers = kwargs['num_layers']
        # 从参数中获取输入形状：行数、列数、通道数
        rows, cols, bands = kwargs['shape']

        # 初始化编码器列表：
        # 每个编码器的下采样率为2^_l，其中_l从0到num_layers-1
        # 例如：第一层下采样率为1（不下采样），第二层为2，第三层为4，以此类推
        self.encoders = nn.ModuleList([
            Encoder(shape=(rows, cols, bands), down_rate=2 ** _l)
            for _l in range(self.num_layers)
        ])

        # 初始化解码器列表：
        # 解码器数量与编码器数量相同
        self.decoders = nn.ModuleList([
            Decoder(shape=(rows, cols, bands))
            for _l in range(self.num_layers)
        ])

    # 前向传播函数
    def forward(self, x):
        # 输入数据形状转换：
        # 输入x形状为(rows, cols, bands)
        # permute(2, 0, 1)转换为(bands, rows, cols)
        # unsqueeze(0)添加批次维度，最终形状为(1, bands, rows, cols)
        x = x.permute(2, 0, 1).unsqueeze(0)

        # 初始化解码列表，用于存储各层解码器的输出
        decoding_list = []

        # ======================== 编码阶段 ========================
        # 初始化编码和为输入x
        encoding_sum = x
        # 遍历所有编码器层
        for _l in range(self.num_layers):
            # 调用第_l个编码器，输入为编码和
            # 返回下采样后的特征output和跳连特征sm
            output, sm = self.encoders[_l](encoding_sum)
            # 将编码器输出添加到解码列表
            decoding_list.append(output)
            # 更新编码和：编码和加上跳连特征sm
            encoding_sum = sm + encoding_sum

        # ======================== 解码阶段 ========================
        # 初始化解码和为与x形状相同的零张量
        decoding_sum = torch.zeros_like(x)
        # 反向遍历解码器层（从最后一层到第一层）
        for _cd in range(self.num_layers - 1, -1, -1):
            # 获取当前解码器对应的编码器输出
            encoder_output = decoding_list[_cd]
            # 调用第_cd个解码器，输入为编码器输出和当前解码和
            decoder_output = self.decoders[_cd](encoder_output, decoding_sum)
            # 更新解码和：解码和加上当前解码器输出
            decoding_sum = decoder_output + decoding_sum
            # 将解码器输出存储到解码列表中
            decoding_list[_cd] = decoder_output

        # ======================== 输出形状转换 ========================
        # 将解码列表中的每个特征图转换回原始输入形状：
        # 1. squeeze(0)移除批次维度，形状变为(bands, rows, cols)
        # 2. permute(1, 2, 0)转换为(rows, cols, bands)
        to_orig_shape = map(
            lambda _x: _x.squeeze(0).permute(1, 2, 0),
            decoding_list
        )
        # 将转换后的结果转换为元组返回
        return tuple(to_orig_shape)

