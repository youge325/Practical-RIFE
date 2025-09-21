import paddle
import numpy as np
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.vision.models as models

# 选择计算设备（优先 GPU）
device = paddle.CUDAPlace(0) if paddle.is_compiled_with_cuda() else paddle.CPUPlace()

# 定义 EPE 损失类（光流端点误差）
class EPE(nn.Layer):
    # 初始化函数
    def __init__(self):
        super(EPE, self).__init__()

    # 前向计算：flow 与 gt 的掩码 L2 误差
    def forward(self, flow, gt, loss_mask):
        # 计算平方误差图
        loss_map = (flow - gt.detach()) ** 2
        # 对通道求和并取平方根得到 L2
        loss_map = (loss_map.sum(1, keepdim=True) + 1e-6) ** 0.5
        # 乘以掩码筛选有效区域
        return (loss_map * loss_mask)

# 定义三值结构相似损失（Ternary Loss）
class Ternary(nn.Layer):
    # 初始化：构造提取局部补丁的 one-hot 卷积核
    def __init__(self):
        super(Ternary, self).__init__()
        patch_size = 7  # 补丁尺寸
        out_channels = patch_size * patch_size  # 输出通道数量
        # 生成单位矩阵并 reshape 成卷积核
        self.w = np.eye(out_channels).reshape((patch_size, patch_size, 1, out_channels))
        # 调整维度顺序以匹配 conv2d 权重格式
        self.w = np.transpose(self.w, (3, 2, 0, 1))
        # 转为张量并放置设备
        self.w = paddle.to_tensor(self.w).astype('float32').to(device)

    # 局部变换：提取补丁并归一化
    def transform(self, img):
        # 使用卷积提取邻域补丁
        patches = F.conv2d(img, self.w, padding=3, bias=None)
        # 与原图差值（中心化）
        transf = patches - img
        # 归一化增强鲁棒性
        transf_norm = transf / paddle.sqrt(0.81 + transf**2)
        return transf_norm

    # 将 RGB 图转为灰度
    def rgb2gray(self, rgb):
        # 拆分 R G B 通道
        r, g, b = rgb[:, 0:1, :, :], rgb[:, 1:2, :, :], rgb[:, 2:3, :, :]
        # 加权求和
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray

    # Hamming 风格距离
    def hamming(self, t1, t2):
        # 计算差平方
        dist = (t1 - t2) ** 2
        # 归一化减弱大值影响
        dist_norm = paddle.mean(dist / (0.1 + dist), axis=1, keepdim=True)
        return dist_norm

    # 生成有效区域掩码
    def valid_mask(self, t, padding):
        # 获取尺寸
        n, _, h, w = t.shape
        # 内部区域为 1
        inner = paddle.ones([n, 1, h - 2 * padding, w - 2 * padding]).astype(t.dtype)
        # 填充恢复原尺寸
        mask = F.pad(inner, [padding] * 4)
        return mask

    # 前向：计算三值距离
    def forward(self, img0, img1):
        # 灰度 + 变换
        img0 = self.transform(self.rgb2gray(img0))
        img1 = self.transform(self.rgb2gray(img1))
        # 距离乘以掩码
        return self.hamming(img0, img1) * self.valid_mask(img0, 1)

# 定义 Sobel 边缘损失类
class SOBEL(nn.Layer):
    # 初始化：构造 Sobel 卷积核
    def __init__(self):
        super(SOBEL, self).__init__()
        # X 方向核
        self.kernelX = paddle.to_tensor([
            [1, 0, -1],
            [2, 0, -2],
            [1, 0, -1],
        ]).astype('float32')
        # Y 方向核（转置）
        self.kernelY = self.kernelX.t()
        # 调整维度为 (1,1,3,3)
        self.kernelX = self.kernelX.unsqueeze(0).unsqueeze(0).to(device)
        self.kernelY = self.kernelY.unsqueeze(0).unsqueeze(0).to(device)

    # 前向：计算边缘差异
    def forward(self, pred, gt):
        # 获取张量维度
        N, C, H, W = pred.shape[0], pred.shape[1], pred.shape[2], pred.shape[3]
        # 拼接预测与真值
        img_stack = paddle.concat([pred.reshape([N*C, 1, H, W]), gt.reshape([N*C, 1, H, W])], 0)
        # 计算 X 方向梯度
        sobel_stack_x = F.conv2d(img_stack, self.kernelX, padding=1)
        # 计算 Y 方向梯度
        sobel_stack_y = F.conv2d(img_stack, self.kernelY, padding=1)
        # 拆分预测与真值 X
        pred_X, gt_X = sobel_stack_x[:N*C], sobel_stack_x[N*C:]
        # 拆分预测与真值 Y
        pred_Y, gt_Y = sobel_stack_y[:N*C], sobel_stack_y[N*C:]
        # 计算 L1 差值
        L1X, L1Y = paddle.abs(pred_X-gt_X), paddle.abs(pred_Y-gt_Y)
        # 汇总损失
        loss = (L1X+L1Y)
        # 返回损失
        return loss

# 均值方差归一化层（仿 transforms.Normalize）
class MeanShift(nn.Conv2D):
    # 初始化：传入均值 std 与输出范围
    def __init__(self, data_mean, data_std, data_range=1, norm=True):
        c = len(data_mean)  # 通道数
        super(MeanShift, self).__init__(c, c, kernel_size=1)
        std = paddle.to_tensor(data_std)  # 标准差张量
        # 初始化权重为单位矩阵
        self.weight.set_value(paddle.eye(c).reshape([c, c, 1, 1]))
        # 如果需要标准化
        if norm:
            # 权重除以标准差
            self.weight.set_value(self.weight / std.reshape([c, 1, 1, 1]))
            # 设置偏置为 -mean * range
            self.bias.set_value(-1 * data_range * paddle.to_tensor(data_mean))
            # 偏置再除以 std
            self.bias.set_value(self.bias / std)
        else:
            # 不标准化时：权重乘以 std
            self.weight.set_value(self.weight * std.reshape([c, 1, 1, 1]))
            # 偏置为 mean * range
            self.bias.set_value(data_range * paddle.to_tensor(data_mean))
        # 不参与梯度训练
        self.stop_gradient = True

# 定义 VGG 感知损失类
class VGGPerceptualLoss(nn.Layer):
    # 初始化：加载预训练 VGG19 并冻结
    def __init__(self, rank=0):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []  # 预留（未使用）
        pretrained = True  # 是否加载预训练
        # 获取 VGG19 特征提取层
        self.vgg_pretrained_features = models.vgg19(pretrained=pretrained).features
        # 定义输入归一化层
        self.normalize = MeanShift([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], norm=True).to(device)
        # 冻结所有参数
        for param in self.parameters():
            param.stop_gradient = True

    # 前向：计算多层特征 L1 感知损失
    def forward(self, X, Y, indices=None):
        # 对输入做 imagenet 归一化
        X = self.normalize(X)
        # 对目标做同样归一化
        Y = self.normalize(Y)
        # 指定使用的层索引
        indices = [2, 7, 12, 21, 30]
        # 每层的权重
        weights = [1.0/2.6, 1.0/4.8, 1.0/3.7, 1.0/5.6, 10/1.5]
        # 指针索引
        k = 0
        # 初始化损失
        loss = 0
        # 遍历特征层
        for i in range(indices[-1]):
            # 前向第 i 层
            X = self.vgg_pretrained_features[i](X)
            # 同层作用于 Y
            Y = self.vgg_pretrained_features[i](Y)
            # 如果该层在选中列表中
            if (i+1) in indices:
                # 计算当前层差异（detach Y）
                loss += weights[k] * (X - Y.detach()).abs().mean() * 0.1
                # 层计数 +1
                k += 1
        # 返回累积损失
        return loss

# 主程序测试入口
if __name__ == '__main__':
    # 构造零张量 img0
    img0 = paddle.zeros([3, 3, 256, 256]).astype('float32').to(device)
    # 构造随机噪声张量 img1
    img1 = paddle.to_tensor(np.random.normal(0, 1, (3, 3, 256, 256))).astype('float32').to(device)
    # 实例化三值损失
    ternary_loss = Ternary()
    # 打印输出尺寸
    print(ternary_loss(img0, img1).shape)