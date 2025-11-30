# 从当前包的coast模块中导入coast模块本身
# 这样可以通过from dataset import coast直接导入coast模块，无需指定完整路径
# 方便其他模块使用coast数据集的功能，如coast.get_data()
from .coast import coast
