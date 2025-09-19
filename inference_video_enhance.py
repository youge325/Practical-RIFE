import os
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
import warnings
import _thread
import skvideo.io
from queue import Queue, Empty
from model.pytorch_msssim import ssim_matlab
# 关闭警告
warnings.filterwarnings("ignore")

def transferAudio(sourceVideo, targetVideo):
    # 导入内部模块
    import shutil
    import moviepy.editor
    # 定义临时音频文件
    tempAudioFileName = "./temp/audio.mkv"
    # 固定进入块
    if True:
        # 检查目录
        if os.path.isdir("temp"):
            # 删除目录
            shutil.rmtree("temp")
        # 创建目录
        os.makedirs("temp")
        # 提取音频
        os.system('ffmpeg -y -i "{}" -c:a copy -vn {}'.format(sourceVideo, tempAudioFileName))
    # 构造无音频名
    targetNoAudio = os.path.splitext(targetVideo)[0] + "_noaudio" + os.path.splitext(targetVideo)[1]
    # 重命名原视频
    os.rename(targetVideo, targetNoAudio)
    # 合并音视频
    os.system('ffmpeg -y -i "{}" -i {} -c copy "{}"'.format(targetNoAudio, tempAudioFileName, targetVideo))
    # 检查文件大小
    if os.path.getsize(targetVideo) == 0:
        # 设置转码输出
        tempAudioFileName = "./temp/audio.m4a"
        # 转码音频
        os.system('ffmpeg -y -i "{}" -c:a aac -b:a 160k -vn {}'.format(sourceVideo, tempAudioFileName))
        # 重新合并
        os.system('ffmpeg -y -i "{}" -i {} -c copy "{}"'.format(targetNoAudio, tempAudioFileName, targetVideo))
        # 检查再次失败
        if (os.path.getsize(targetVideo) == 0):
            # 回退原视频
            os.rename(targetNoAudio, targetVideo)
            # 打印失败
            print("Audio transfer failed. Interpolated video will have no audio")
        else:
            # 打印转码成功
            print("Lossless audio transfer failed. Audio was transcoded to AAC (M4A) instead.")
            # 删除无音频文件
            os.remove(targetNoAudio)
    else:
        # 删除临时无音频
        os.remove(targetNoAudio)
    # 删除临时目录
    shutil.rmtree("temp")

# 创建解析器
parser = argparse.ArgumentParser(description='Video SR')
# 添加视频参数
parser.add_argument('--video', dest='video', type=str, default=None)
# 添加输出参数
parser.add_argument('--output', dest='output', type=str, default=None)
# 添加图片参数
parser.add_argument('--img', dest='img', type=str, default=None)
# 添加模型目录
parser.add_argument('--model', dest='modelDir', type=str, default='train_log_SAFA', help='directory with trained model files')
# 添加 fp16 参数
parser.add_argument('--fp16', dest='fp16', action='store_true', help='fp16 mode for faster and more lightweight inference on cards with Tensor Cores')
# 添加 png 参数
parser.add_argument('--png', dest='png', action='store_true', help='whether to vid_out png format vid_outs')
# 添加扩展名参数
parser.add_argument('--ext', dest='ext', type=str, default='mp4', help='vid_out video extension')

# 解析参数
args = parser.parse_args()
# 校验输入
assert (not args.video is None or not args.img is None)
# 判断图片模式
if not args.img is None:
    # 设置 png 输出
    args.png = True

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 关闭梯度
torch.set_grad_enabled(False)
# 判断 CUDA
if torch.cuda.is_available():
    # 启用 cudnn
    torch.backends.cudnn.enabled = True
    # 启用 benchmark
    torch.backends.cudnn.benchmark = True
    # 判断 fp16
    if(args.fp16):
        # 打印设置
        print('set fp16')
        # 设置默认张量
        torch.set_default_tensor_type(torch.cuda.HalfTensor)

# 尝试导入模型
try:
    # 导入模型
    from train_log_SAFA.model import Model
except:
    # 打印下载提示
    print("Please download our model from model list")
# 实例化模型
model = Model()
# 绑定设备
model.device()
# 加载权重
model.load_model(args.modelDir)
# 打印加载
print("Loaded SAFA model.")
# 切换推理
model.eval()

# 判断视频输入
if not args.video is None:
    # 打开视频
    videoCapture = cv2.VideoCapture(args.video)
    # 获取 fps
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    # 获取帧数
    tot_frame = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
    # 释放句柄
    videoCapture.release()
    # 标记 fps 状态
    fpsNotAssigned = True
    # 创建生成器
    videogen = skvideo.io.vreader(args.video)
    # 读取第一帧
    lastframe = next(videogen)
    # 定义编码
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    # 分离路径
    video_path_wo_ext, ext = os.path.splitext(args.video)
    # 判断是否合并音频
    if args.png == False and fpsNotAssigned == True:
        # 打印将合并
        print("The audio will be merged after interpolation process")
    else:
        # 打印不合并
        print("Will not merge audio because using png or fps flag!")
else:
    # 初始化帧名列表
    videogen = []
    # 遍历目录
    for f in os.listdir(args.img):
        # 匹配 png
        if 'png' in f:
            # 添加列表
            videogen.append(f)
    # 统计帧数
    tot_frame = len(videogen)
    # 排序帧名
    videogen.sort(key= lambda x:int(x[:-4]))
    # 读取首帧
    lastframe = cv2.imread(os.path.join(args.img, videogen[0]), cv2.IMREAD_UNCHANGED)[:, :, ::-1].copy()
    # 截取剩余
    videogen = videogen[1:]

# 解析尺寸
h, w, _ = lastframe.shape
# 初始化输出名
vid_out_name = None
# 初始化 writer
vid_out = None
# 判断 png 输出
if args.png:
    # 检查目录
    if not os.path.exists('vid_out'):
        # 创建目录
        os.mkdir('vid_out')
else:
    # 判断指定输出
    if args.output is not None:
        # 使用指定
        vid_out_name = args.output
    else:
        # 自动拼接名
        vid_out_name = '{}_2X{}'.format(video_path_wo_ext, ext)
    # 创建写入器
    vid_out = cv2.VideoWriter(vid_out_name, fourcc, fps, (w, h))

def clear_write_buffer(user_args, write_buffer):
    # 初始化计数
    cnt = 0
    # 循环写出
    while True:
        # 获取元素
        item = write_buffer.get()
        # 判断结束
        if item is None:
            # 结束循环
            break
        # 判断 png
        if user_args.png:
            # 写 PNG 文件
            cv2.imwrite('vid_out/{:0>7d}.png'.format(cnt), item[:, :, ::-1])
            # 计数递增
            cnt += 1
        else:
            # 写视频帧
            vid_out.write(item[:, :, ::-1])

def build_read_buffer(user_args, read_buffer, videogen):
    # 遍历源帧
    for frame in videogen:
        # 判断图片模式
        if not user_args.img is None:
            # 读取图片帧
            frame = cv2.imread(os.path.join(user_args.img, frame), cv2.IMREAD_UNCHANGED)[:, :, ::-1].copy()
        # 推入队列
        read_buffer.put(frame)
    # 推入结束
    read_buffer.put(None)

def pad_image(img):
    # 判断半精度
    if(args.fp16):
        # 反射填充并半精度
        return F.pad(img, padding, mode='reflect').half()
    else:
        # 反射填充返回
        return F.pad(img, padding, mode='reflect')

# 基础对齐块
tmp = 64
# 计算填充高
ph = ((h - 1) // tmp + 1) * tmp
# 计算填充宽
pw = ((w - 1) // tmp + 1) * tmp
# 生成 padding
padding = (0, pw - w, 0, ph - h)
# 创建进度条
pbar = tqdm(total=tot_frame)
# 创建写队列
write_buffer = Queue(maxsize=500)
# 创建读队列
read_buffer = Queue(maxsize=500)
# 启动读线程
_thread.start_new_thread(build_read_buffer, (args, read_buffer, videogen))
# 启动写线程
_thread.start_new_thread(clear_write_buffer, (args, write_buffer))

# 主循环
while True:
    # 读取一帧
    frame = read_buffer.get()
    # 判断结束
    if frame is None:
        # 退出循环
        break
    # 构造上一帧张量
    I0 = pad_image(torch.from_numpy(np.transpose(lastframe, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.)
    # 构造当前帧张量
    I1 = pad_image(torch.from_numpy(np.transpose(frame, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.)
    # 下采样上一帧
    I0_small = F.interpolate(I0, (32, 32), mode='bilinear', align_corners=False)
    # 下采样当前帧
    I1_small = F.interpolate(I1, (32, 32), mode='bilinear', align_corners=False)
    # 计算结构相似度
    ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])
    # 分支判断
    if ssim < 0.2:
        # 分离增强
        out = [model.inference(I0, I0, [0])[0], model.inference(I1, I1, [0])[0]]
    else:
        # 双帧增强
        out = model.inference(I0, I1, [0, 1])
    # 校验长度
    assert(len(out) == 2)
    # 写第一输出
    write_buffer.put((out[0][0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w])
    # 写第二输出
    write_buffer.put((out[1][0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w])
    # 更新上一帧
    lastframe = read_buffer.get()
    # 判断结束
    if lastframe is None:
        # 退出循环
        break
    # 更新进度
    pbar.update(2)

# 导入时间
import time
# 等待写队列
while(not write_buffer.empty()):
    # 休眠等待
    time.sleep(0.1)
# 关闭进度条
pbar.close()
# 关闭 writer
if not vid_out is None:
    # 释放资源
    vid_out.release()

# 判断复制音频
if args.png == False and fpsNotAssigned == True and not args.video is None:
    # 捕获异常
    try:
        # 复制音频
        transferAudio(args.video, vid_out_name)
    except:
        # 打印失败
        print("Audio transfer failed. Interpolated video will have no audio")
        # 构造无音频名
        targetNoAudio = os.path.splitext(vid_out_name)[0] + "_noaudio" + os.path.splitext(vid_out_name)[1]
        # 回退重命名
        os.rename(targetNoAudio, vid_out_name)
