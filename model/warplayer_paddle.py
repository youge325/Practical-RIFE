import paddle
import paddle.nn as nn

# 选择计算设备（GPU 优先）
device = paddle.CUDAPlace(0) if paddle.is_compiled_with_cuda() else paddle.CPUPlace()
# 初始化/缓存不同尺寸与设备的网格字典
backwarp_tenGrid = {}

# 定义 warp 函数：利用光流对输入进行反向采样（实际是 forward warp 的 backward 实现）
def warp(tenInput, tenFlow):
    # 生成当前光流张量对应的 key（包含设备与尺寸）
    k = (str(tenFlow.place), str(tenFlow.shape))
    # 若该尺寸光流的采样基础网格尚未缓存则创建
    if k not in backwarp_tenGrid:
        # 构建水平方向标准化坐标序列并扩展到 (N,1,H,W)
        tenHorizontal = paddle.linspace(-1.0, 1.0, tenFlow.shape[3]).reshape(
            [1, 1, 1, tenFlow.shape[3]]).expand([tenFlow.shape[0], -1, tenFlow.shape[2], -1])
        # 构建垂直方向标准化坐标序列并扩展到 (N,1,H,W)
        tenVertical = paddle.linspace(-1.0, 1.0, tenFlow.shape[2]).reshape(
            [1, 1, tenFlow.shape[2], 1]).expand([tenFlow.shape[0], -1, -1, tenFlow.shape[3]])
        # 将两个方向坐标拼接成基础网格并缓存
        backwarp_tenGrid[k] = paddle.concat(
            [tenHorizontal, tenVertical], 1).to(device)

    # 将光流的像素位移转换为 [-1,1] 归一化坐标增量（按宽高各自尺度归一化）
    tenFlow = paddle.concat([
        tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),  # 归一化水平位移
        tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)   # 归一化垂直位移
    ], 1)

    # 叠加基础网格与位移获得采样网格 (N,H,W,2)
    g = (backwarp_tenGrid[k] + tenFlow).transpose([0, 2, 3, 1])
    # 使用 grid_sample 进行双线性采样并在越界时使用边界像素填充
    return paddle.nn.functional.grid_sample(
        x=tenInput,
        grid=g,
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    )