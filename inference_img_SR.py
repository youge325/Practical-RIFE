import os
import cv2
import torch
import argparse
from torch.nn import functional as F
import warnings
# 关闭警告
warnings.filterwarnings("ignore")

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 关闭梯度
torch.set_grad_enabled(False)
# 启用 cuDNN
if torch.cuda.is_available():
    # 开启 cudnn
    torch.backends.cudnn.enabled = True
    # 开启 benchmark
    torch.backends.cudnn.benchmark = True

# 创建解析器
parser = argparse.ArgumentParser(description='STVSR for a pair of images')
# 添加图像参数
parser.add_argument('--img', dest='img', nargs=2, required=True)
# 添加指数参数
parser.add_argument('--exp', default=2, type=int)
# 添加 ratio 参数
parser.add_argument('--ratio', default=0, type=float, help='inference ratio between two images with 0 - 1 range')
# 添加模型目录
parser.add_argument('--model', dest='modelDir', type=str, default='train_log', help='directory with trained model files')

# 解析参数
args = parser.parse_args()

# 导入模型
from train_log.model import Model
# 实例化模型
model = Model()
# 绑定设备
model.device()
# 加载权重
model.load_model('train_log')
# 设置评估
model.eval()

# 判断 EXR 输入
if args.img[0].endswith('.exr') and args.img[1].endswith('.exr'):
    # 读取 EXR0
    img0 = cv2.imread(args.img[0], cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
    # 读取 EXR1
    img1 = cv2.imread(args.img[1], cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
    # 上采样 EXR0
    img0 = cv2.resize(img0, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    # 上采样 EXR1
    img1 = cv2.resize(img1, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    # 转张量0
    img0 = (torch.tensor(img0.transpose(2, 0, 1)).to(device)).unsqueeze(0)
    # 转张量1
    img1 = (torch.tensor(img1.transpose(2, 0, 1)).to(device)).unsqueeze(0)
else:
    # 读取普通0
    img0 = cv2.imread(args.img[0], cv2.IMREAD_UNCHANGED)
    # 读取普通1
    img1 = cv2.imread(args.img[1], cv2.IMREAD_UNCHANGED)
    # 上采样普通0
    img0 = cv2.resize(img0, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    # 上采样普通1
    img1 = cv2.resize(img1, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    # 归一化张量0
    img0 = (torch.tensor(img0.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
    # 归一化张量1
    img1 = (torch.tensor(img1.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)

# 读取尺寸
n, c, h, w = img0.shape
# 计算对齐高
ph = ((h - 1) // 32 + 1) * 32
# 计算对齐宽
pw = ((w - 1) // 32 + 1) * 32
# 计算 padding
padding = (0, pw - w, 0, ph - h)
# 填充图像0
img0 = F.pad(img0, padding)
# 填充图像1
img1 = F.pad(img1, padding)

# 判断 ratio 分支
if args.ratio:
    # 打印 ratio
    print('ratio={}'.format(args.ratio))
    # 单点推理
    img_list = model.inference(img0, img1, timestep=args.ratio)
else:
    # 计算中间帧数量
    n = 2 ** args.exp - 1
    # 初始化时间表
    time_list = [0]
    # 生成时间序列
    for i in range(n):
        # 添加时间点
        time_list.append((i+1) * 1. / (n+1))
    # 添加终点
    time_list.append(1)
    # 打印时间表
    print(time_list)
    # 序列推理
    img_list = model.inference(img0, img1, timestep=time_list)

# 检查输出目录
if not os.path.exists('output'):
    # 创建目录
    os.mkdir('output')
# 遍历结果
for i in range(len(img_list)):
    # 判断 EXR 输出
    if args.img[0].endswith('.exr') and args.img[1].endswith('.exr'):
        # 写 EXR 文件
        cv2.imwrite('output/img{}.exr'.format(i), (img_list[i][0]).cpu().numpy().transpose(1, 2, 0)[:h, :w], [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF])
    else:
        # 写 PNG 文件
        cv2.imwrite('output/img{}.png'.format(i), (img_list[i][0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w])
