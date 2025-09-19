from math import exp  # 指数函数
import torch  # 导入 PyTorch
import torch.nn.functional as F  # 导入函数式 API
import torch.nn as nn  # 导入神经网络模块

# 选择运行设备（优先 GPU）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 生成一维高斯核函数（返回长度为 window_size 的向量）
def gaussian(window_size: int, sigma: float):
    # 生成高斯权重列表
    gauss = torch.tensor([exp(-(x - window_size // 2) ** 2 / (2 * sigma ** 2)) for x in range(window_size)], dtype=torch.float32)
    # 归一化权重使其和为 1
    return gauss / gauss.sum()

# 创建二维高斯窗口（返回形状 [channel,1,H,W]）
def create_window(window_size: int, channel: int = 1):
    # 生成一维高斯列向量
    _1d = gaussian(window_size, 1.5).unsqueeze(1)
    # 通过外积得到二维高斯矩阵
    _2d = _1d @ _1d.t()
    # 转为 float 并添加 batch/channel 维度
    _2d = _2d.float().unsqueeze(0).unsqueeze(0).to(device)
    # 扩展到指定通道数
    window = _2d.expand(channel, 1, window_size, window_size).contiguous()
    # 返回窗口
    return window

# 创建三维高斯窗口（主要用于 3D SSIM 实验）
def create_window_3d(window_size: int, channel: int = 1):
    # 生成一维高斯列向量
    _1d = gaussian(window_size, 1.5).unsqueeze(1)
    # 生成二维高斯矩阵
    _2d = _1d @ _1d.t()
    # 生成三维高斯体（再与 1D 向量做外积）
    _3d = _2d.unsqueeze(2) @ _1d.t()
    # 扩展为 [1,channel,D,H,W]
    window = _3d.expand(1, channel, window_size, window_size, window_size).contiguous().to(device)
    # 返回 3D 窗口
    return window

# 单尺度 SSIM 计算函数
def ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, window=None, size_average: bool = True, full: bool = False, val_range=None):
    # 若未提供动态范围则根据像素值推断
    if val_range is None:
        # 推断最大值（>128 认为是 0~255 范围）
        max_val = 255 if torch.max(img1) > 128 else 1
        # 推断最小值（<-0.5 认为下界为 -1）
        min_val = -1 if torch.min(img1) < -0.5 else 0
        # 计算动态范围 L
        L = max_val - min_val
    else:
        # 使用指定动态范围
        L = val_range
    # 卷积 padding 大小（这里为 0）
    padd = 0
    # 获取输入尺寸（N,C,H,W）
    (_, channel, height, width) = img1.size()
    # 若窗口为空则构建窗口（窗口不应大于图像尺寸）
    if window is None:
        # 实际窗口尺寸取 min
        real_size = min(window_size, height, width)
        # 创建窗口并放到与输入相同设备
        window = create_window(real_size, channel=channel).to(img1.device)
    # 计算局部均值 mu1
    mu1 = F.conv2d(F.pad(img1, (5, 5, 5, 5), mode='replicate'), window, padding=padd, groups=channel)
    # 计算局部均值 mu2
    mu2 = F.conv2d(F.pad(img2, (5, 5, 5, 5), mode='replicate'), window, padding=padd, groups=channel)
    # 计算 mu1 平方
    mu1_sq = mu1.pow(2)
    # 计算 mu2 平方
    mu2_sq = mu2.pow(2)
    # 计算 mu1 * mu2
    mu1_mu2 = mu1 * mu2
    # 计算局部方差 sigma1^2
    sigma1_sq = F.conv2d(F.pad(img1 * img1, (5, 5, 5, 5), 'replicate'), window, padding=padd, groups=channel) - mu1_sq
    # 计算局部方差 sigma2^2
    sigma2_sq = F.conv2d(F.pad(img2 * img2, (5, 5, 5, 5), 'replicate'), window, padding=padd, groups=channel) - mu2_sq
    # 计算协方差 sigma12
    sigma12 = F.conv2d(F.pad(img1 * img2, (5, 5, 5, 5), 'replicate'), window, padding=padd, groups=channel) - mu1_mu2
    # 定义常数 C1
    C1 = (0.01 * L) ** 2
    # 定义常数 C2
    C2 = (0.03 * L) ** 2
    # 计算对比度分量分子 v1
    v1 = 2.0 * sigma12 + C2
    # 计算对比度分量分母 v2
    v2 = sigma1_sq + sigma2_sq + C2
    # 计算 cs（对比度敏感度项）
    cs = torch.mean(v1 / v2)
    # 计算 SSIM map
    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)
    # 若需要对所有空间位置求平均
    if size_average:
        # 取全局均值
        ret = ssim_map.mean()
    else:
        # 分通道分别聚合（保持 batch）
        ret = ssim_map.mean(1).mean(1).mean(1)
    # 如果需要返回 cs
    if full:
        # 返回 (ssim, cs)
        return ret, cs
    # 否则仅返回 ssim
    return ret

# MATLAB 风格 3D SSIM（将输入视为体数据）
def ssim_matlab(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, window=None, size_average: bool = True, full: bool = False, val_range=None):
    # 若未提供动态范围则自动推断
    if val_range is None:
        # 推断最大值
        max_val = 255 if torch.max(img1) > 128 else 1
        # 推断最小值
        min_val = -1 if torch.min(img1) < -0.5 else 0
        # 计算动态范围 L
        L = max_val - min_val
    else:
        # 使用传入动态范围
        L = val_range
    # 卷积 padding
    padd = 0
    # 获取尺寸（忽略通道）
    (_, _, height, width) = img1.size()
    # 若未提供窗口则创建 3D 窗口
    if window is None:
        # 真实窗口尺寸
        real_size = min(window_size, height, width)
        # 创建 3D 高斯窗口
        window = create_window_3d(real_size, channel=1).to(img1.device)
    # 扩展一维作为“深度”维（模拟体数据）
    img1 = img1.unsqueeze(1)
    # 同样处理第二张图
    img2 = img2.unsqueeze(1)
    # 计算三维均值 mu1
    mu1 = F.conv3d(F.pad(img1, (5, 5, 5, 5, 5, 5), mode='replicate'), window, padding=padd, groups=1)
    # 计算三维均值 mu2
    mu2 = F.conv3d(F.pad(img2, (5, 5, 5, 5, 5, 5), mode='replicate'), window, padding=padd, groups=1)
    # 计算 mu1 平方
    mu1_sq = mu1.pow(2)
    # 计算 mu2 平方
    mu2_sq = mu2.pow(2)
    # 计算 mu1*mu2
    mu1_mu2 = mu1 * mu2
    # 计算方差 sigma1^2
    sigma1_sq = F.conv3d(F.pad(img1 * img1, (5, 5, 5, 5, 5, 5), 'replicate'), window, padding=padd, groups=1) - mu1_sq
    # 计算方差 sigma2^2
    sigma2_sq = F.conv3d(F.pad(img2 * img2, (5, 5, 5, 5, 5, 5), 'replicate'), window, padding=padd, groups=1) - mu2_sq
    # 计算协方差 sigma12
    sigma12 = F.conv3d(F.pad(img1 * img2, (5, 5, 5, 5, 5, 5), 'replicate'), window, padding=padd, groups=1) - mu1_mu2
    # 定义常数 C1
    C1 = (0.01 * L) ** 2
    # 定义常数 C2
    C2 = (0.03 * L) ** 2
    # 计算 v1
    v1 = 2.0 * sigma12 + C2
    # 计算 v2
    v2 = sigma1_sq + sigma2_sq + C2
    # 计算 cs
    cs = torch.mean(v1 / v2)
    # 计算 SSIM map
    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)
    # 聚合方式
    if size_average:
        # 全局均值
        ret = ssim_map.mean()
    else:
        # 局部聚合
        ret = ssim_map.mean(1).mean(1).mean(1)
    # 是否需要返回 cs
    if full:
        # 返回 (ssim, cs)
        return ret, cs
    # 否则仅返回 ssim
    return ret

# 多尺度 SSIM 计算函数
def msssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, size_average: bool = True, val_range=None, normalize: bool = False):
    # 定义多尺度权重
    weights = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=torch.float32).to(img1.device)
    # 计算层级数量
    levels = weights.numel()
    # 初始化 SSIM 列表
    mssim_list = []
    # 初始化 CS 列表
    mcs_list = []
    # 遍历每个层级
    for _ in range(levels):
        # 计算当前尺度的 ssim 与 cs
        sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average, full=True, val_range=val_range)
        # 追加 ssim 值
        mssim_list.append(sim)
        # 追加 cs 值
        mcs_list.append(cs)
        # 平均池化下采样 img1
        img1 = F.avg_pool2d(img1, (2, 2))
        # 平均池化下采样 img2
        img2 = F.avg_pool2d(img2, (2, 2))
    # 将 ssim 列表堆叠为张量
    mssim_t = torch.stack(mssim_list)
    # 将 cs 列表堆叠为张量
    mcs_t = torch.stack(mcs_list)
    # 如果需要归一化到 [0,1]
    if normalize:
        # 归一化 ssim
        mssim_t = (mssim_t + 1) / 2
        # 归一化 cs
        mcs_t = (mcs_t + 1) / 2
    # 计算 mcs 的加权幂（除最后尺度）
    pow1 = mcs_t ** weights
    # 计算最后尺度 ssim 幂
    pow2 = mssim_t ** weights
    # 组合输出（忽略最后尺度的 cs）
    output = torch.prod(pow1[:-1] * pow2[-1])
    # 返回多尺度 SSIM
    return output

# 旧接口函数式封装（保持向后兼容）
def MSSSIM(window_size=11, size_average=True, channel=3):
    # 定义内部闭包函数
    def _fn(img1, img2):
        # 调用 msssim 主函数
        return msssim(img1, img2, window_size=window_size, size_average=size_average)
    # 返回闭包
    return _fn

# 面向对象 SSIM 封装（包含窗口缓存）
class SSIM(nn.Module):
    # 初始化函数
    def __init__(self, window_size=11, size_average=True, val_range=None):
        # 调用父类初始化
        super().__init__()
        # 保存窗口大小
        self.window_size = window_size
        # 保存是否求平均
        self.size_average = size_average
        # 保存值域范围
        self.val_range = val_range
        # 默认通道数量
        self.channel = 3
        # 初始化窗口缓存
        self.window = create_window(window_size, channel=self.channel)
    # 前向计算函数
    def forward(self, img1, img2):
        # 解析输入通道数
        _, c, _, _ = img1.size()
        # 如果缓存匹配则直接使用
        if c == self.channel and self.window.dtype == img1.dtype:
            # 使用已有窗口
            window = self.window
        else:
            # 创建新的窗口
            window = create_window(self.window_size, c).to(img1.device).type(img1.dtype)
            # 更新窗口缓存
            self.window = window
            # 更新通道记录
            self.channel = c
        # 计算单尺度 ssim
        _ssim = ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)
        # 转换为差异形式 dssim（范围约 0~1）
        dssim = (1 - _ssim) / 2
        # 返回 dssim
        return dssim
