import os
import cv2
import torch
import argparse
from torch.nn import functional as F
import warnings
# 关闭所有警告
warnings.filterwarnings("ignore")

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 关闭梯度
torch.set_grad_enabled(False)
# 若可用启用 cuDNN
if torch.cuda.is_available():
    # 打开 cuDNN
    torch.backends.cudnn.enabled = True
    # 打开 benchmark
    torch.backends.cudnn.benchmark = True

# 创建解析器
parser = argparse.ArgumentParser(description='Interpolation for a pair of images')
# 添加图像参数
parser.add_argument('--img', dest='img', nargs=2, required=True)
# 添加指数参数
parser.add_argument('--exp', default=4, type=int)
# 添加 ratio 参数
parser.add_argument('--ratio', default=0, type=float, help='inference ratio between two images with 0 - 1 range')
# 添加 ratio 阈值
parser.add_argument('--rthreshold', default=0.02, type=float, help='returns image when actual ratio falls in given range threshold')
# 添加最大循环
parser.add_argument('--rmaxcycles', default=8, type=int, help='limit max number of bisectional cycles')
# 添加模型目录
parser.add_argument('--model', dest='modelDir', type=str, default='train_log', help='directory with trained model files')

# 解析参数
args = parser.parse_args()

# 尝试加载模型
try:
    # 内层尝试 v2
    try:
        # 导入 v2 模型
        from model.RIFE_HDv2 import Model
        # 实例化模型
        model = Model()
        # 加载权重
        model.load_model(args.modelDir, -1)
        # 打印加载信息
        print("Loaded v2.x HD model.")
    except:
        # 回退 v3
        from train_log.RIFE_HDv3 import Model
        # 实例化 v3
        model = Model()
        # 加载权重
        model.load_model(args.modelDir, -1)
        # 打印加载信息
        print("Loaded v3.x HD model.")
except:
    # 继续回退 v1
    from model.RIFE_HD import Model
    # 实例化 v1
    model = Model()
    # 加载权重
    model.load_model(args.modelDir, -1)
    # 打印加载信息
    print("Loaded v1.x HD model")
# 检查版本属性
if not hasattr(model, 'version'):
    # 设置默认版本
    model.version = 0
# 设置评估模式
model.eval()
# 绑定设备
model.device()

# 判断是否为 EXR
if args.img[0].endswith('.exr') and args.img[1].endswith('.exr'):
    # 读取 EXR 图像0
    img0 = cv2.imread(args.img[0], cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
    # 读取 EXR 图像1
    img1 = cv2.imread(args.img[1], cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
    # 转张量图像0
    img0 = (torch.tensor(img0.transpose(2, 0, 1)).to(device)).unsqueeze(0)
    # 转张量图像1
    img1 = (torch.tensor(img1.transpose(2, 0, 1)).to(device)).unsqueeze(0)
else:
    # 读取普通图像0
    img0 = cv2.imread(args.img[0], cv2.IMREAD_UNCHANGED)
    # 读取普通图像1
    img1 = cv2.imread(args.img[1], cv2.IMREAD_UNCHANGED)
    # 缩放图像0
    img0 = cv2.resize(img0, (448, 256))
    # 缩放图像1
    img1 = cv2.resize(img1, (448, 256))
    # 归一化并转张量0
    img0 = (torch.tensor(img0.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
    # 归一化并转张量1
    img1 = (torch.tensor(img1.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)

# 读取形状信息
n, c, h, w = img0.shape
# 计算填充高度
ph = ((h - 1) // 64 + 1) * 64
# 计算填充宽度
pw = ((w - 1) // 64 + 1) * 64
# 计算 padding 元组
padding = (0, pw - w, 0, ph - h)
# 填充图像0
img0 = F.pad(img0, padding)
# 填充图像1
img1 = F.pad(img1, padding)

# 判断 ratio 分支
if args.ratio:
    # 大版本判断
    if model.version >= 3.9:
        # 直接推理指定比例
        img_list = [img0, model.inference(img0, img1, args.ratio), img1]
    else:
        # 初始化左界比例
        img0_ratio = 0.0
        # 初始化右界比例
        img1_ratio = 1.0
        # 判断接近左端
        if args.ratio <= img0_ratio + args.rthreshold / 2:
            # 取左端帧
            middle = img0
        # 判断接近右端
        elif args.ratio >= img1_ratio - args.rthreshold / 2:
            # 取右端帧
            middle = img1
        else:
            # 复制左帧
            tmp_img0 = img0
            # 复制右帧
            tmp_img1 = img1
            # 遍历二分次数
            for inference_cycle in range(args.rmaxcycles):
                # 推理中点
                middle = model.inference(tmp_img0, tmp_img1)
                # 计算中点比例
                middle_ratio = ( img0_ratio + img1_ratio ) / 2
                # 判断落入阈值
                if args.ratio - (args.rthreshold / 2) <= middle_ratio <= args.ratio + (args.rthreshold / 2):
                    # 跳出循环
                    break
                # 判断目标在右侧
                if args.ratio > middle_ratio:
                    # 更新左帧
                    tmp_img0 = middle
                    # 更新左比例
                    img0_ratio = middle_ratio
                else:
                    # 更新右帧
                    tmp_img1 = middle
                    # 更新右比例
                    img1_ratio = middle_ratio
    # 初始化输出列表
    img_list = [img0]
    # 添加中间帧
    img_list.append(middle)
    # 添加终止帧
    img_list.append(img1)
else:
    # 未指定 ratio 分支
    if model.version >= 3.9:
        # 初始化列表
        img_list = [img0]
        # 计算采样数
        n = 2 ** args.exp
        # 遍历生成中间帧
        for i in range(n-1):
            # 推理第 i 帧
            img_list.append(model.inference(img0, img1, (i+1) * 1. / n))
        # 追加终止帧
        img_list.append(img1)
    else:
        # 初始化两端
        img_list = [img0, img1]
        # 多轮倍增
        for i in range(args.exp):
            # 临时列表
            tmp = []
            # 遍历间隔
            for j in range(len(img_list) - 1):
                # 推理中间帧
                mid = model.inference(img_list[j], img_list[j + 1])
                # 添加左帧
                tmp.append(img_list[j])
                # 添加中间
                tmp.append(mid)
            # 添加尾帧
            tmp.append(img1)
            # 覆盖列表
            img_list = tmp

# 判断输出目录
if not os.path.exists('output'):
    # 创建输出目录
    os.mkdir('output')
# 遍历输出序列
for i in range(len(img_list)):
    # 判断 EXR 输出
    if args.img[0].endswith('.exr') and args.img[1].endswith('.exr'):
        # 保存 EXR 文件
        cv2.imwrite('output/img{}.exr'.format(i), (img_list[i][0]).cpu().numpy().transpose(1, 2, 0)[:h, :w], [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF])
    else:
        # 保存 PNG 文件
        cv2.imwrite('output/img{}.png'.format(i), (img_list[i][0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w])
