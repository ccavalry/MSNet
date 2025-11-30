# 定义异常检测函数
# 输入：
#   x: 输入数据，形状为(rows, cols, bands)的PyTorch张量
#   decoder_outputs: 解码器输出列表，每个元素是形状为(rows, cols, bands)的PyTorch张量
# 返回：
#   dm: 异常检测图，形状为(rows, cols)的PyTorch张量，值越大表示异常程度越高

def detect(x, decoder_outputs):
    # 获取第一个解码器输出（主输出），形状为(rows, cols, bands)
    y = decoder_outputs[0]
    # 计算输入x与输出y的差异（重建误差），并分离梯度（不参与反向传播）
    dm = (x - y).detach()
    # 计算重建误差的平方，增强异常区域的对比度
    dm = dm ** 2
    # 沿波段维度（第2维）求和，得到每个像素的总重建误差，形状变为(rows, cols)
    dm = dm.sum(2)

    # 返回异常检测图
    return dm
