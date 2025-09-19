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
    # 导入内部库
    import shutil
    import moviepy.editor
    # 定义临时音频
    tempAudioFileName = "./temp/audio.mkv"
    # 条件恒真块
    if True:
        # 判断 temp 目录
        if os.path.isdir("temp"):
            # 删除旧目录
            shutil.rmtree("temp")
        # 新建目录
        os.makedirs("temp")
        # 拷贝音频流
        os.system('ffmpeg -y -i "{}" -c:a copy -vn {}'.format(sourceVideo, tempAudioFileName))
    # 生成无音频名
    targetNoAudio = os.path.splitext(targetVideo)[0] + "_noaudio" + os.path.splitext(targetVideo)[1]
    # 重命名原视频
    os.rename(targetVideo, targetNoAudio)
    # 合并音视频
    os.system('ffmpeg -y -i "{}" -i {} -c copy "{}"'.format(targetNoAudio, tempAudioFileName, targetVideo))
    # 检查失败
    if os.path.getsize(targetVideo) == 0:
        # 改用 AAC
        tempAudioFileName = "./temp/audio.m4a"
        # 转码音频
        os.system('ffmpeg -y -i "{}" -c:a aac -b:a 160k -vn {}'.format(sourceVideo, tempAudioFileName))
        # 再次合并
        os.system('ffmpeg -y -i "{}" -i {} -c copy "{}"'.format(targetNoAudio, tempAudioFileName, targetVideo))
        # 再次检查
        if (os.path.getsize(targetVideo) == 0):
            # 回退重命名
            os.rename(targetNoAudio, targetVideo)
            # 打印失败
            print("Audio transfer failed. Interpolated video will have no audio")
        else:
            # 打印转码
            print("Lossless audio transfer failed. Audio was transcoded to AAC (M4A) instead.")
            # 删除临时视频
            os.remove(targetNoAudio)
    else:
        # 删除无音频暂存
        os.remove(targetNoAudio)
    # 删除临时目录
    shutil.rmtree("temp")

# 创建解析器
parser = argparse.ArgumentParser(description='Interpolation for a pair of images')
# 添加视频参数
parser.add_argument('--video', dest='video', type=str, default=None)
# 添加输出参数
parser.add_argument('--output', dest='output', type=str, default=None)
# 添加图片目录
parser.add_argument('--img', dest='img', type=str, default=None)
# 添加拼接参数
parser.add_argument('--montage', dest='montage', action='store_true', help='montage origin video')
# 添加模型目录
parser.add_argument('--model', dest='modelDir', type=str, default='train_log', help='directory with trained model files')
# 添加 fp16 参数
parser.add_argument('--fp16', dest='fp16', action='store_true', help='fp16 mode for faster and more lightweight inference on cards with Tensor Cores')
# 添加 UHD 参数
parser.add_argument('--UHD', dest='UHD', action='store_true', help='support 4k video')
# 添加 scale 参数
parser.add_argument('--scale', dest='scale', type=float, default=1.0, help='Try scale=0.5 for 4k video')
# 添加 skip 参数
parser.add_argument('--skip', dest='skip', action='store_true', help='whether to remove static frames before processing')
# 添加 fps 参数
parser.add_argument('--fps', dest='fps', type=int, default=None)
# 添加 png 参数
parser.add_argument('--png', dest='png', action='store_true', help='whether to vid_out png format vid_outs')
# 添加扩展名
parser.add_argument('--ext', dest='ext', type=str, default='mp4', help='vid_out video extension')
# 添加 exp 参数
parser.add_argument('--exp', dest='exp', type=int, default=1)
# 添加 multi 参数
parser.add_argument('--multi', dest='multi', type=int, default=2)

# 解析参数
args = parser.parse_args()
# 判断 exp
if args.exp != 1:
    # 计算 multi
    args.multi = (2 ** args.exp)
# 校验输入来源
assert (not args.video is None or not args.img is None)
# 判断 skip
if args.skip:
    # 提示弃用
    print("skip flag is abandoned, please refer to issue #207.")
# UHD 逻辑
if args.UHD and args.scale==1.0:
    # 调整 scale
    args.scale = 0.5
# 校验 scale
assert args.scale in [0.25, 0.5, 1.0, 2.0, 4.0]
# 图片模式强制 png
if not args.img is None:
    # 设置 png
    args.png = True

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 关闭梯度
torch.set_grad_enabled(False)
# 启用 cudnn
if torch.cuda.is_available():
    # 启用 cudnn
    torch.backends.cudnn.enabled = True
    # 启用 benchmark
    torch.backends.cudnn.benchmark = True
    # 半精度判断
    if(args.fp16):
        # 设置默认半精度
        torch.set_default_tensor_type(torch.cuda.HalfTensor)

# 导入模型
from train_log.RIFE_HDv3 import Model
# 实例化模型
model = Model()
# 检查版本
if not hasattr(model, 'version'):
    # 设置默认版本
    model.version = 0
# 加载权重
model.load_model(args.modelDir, -1)
# 打印加载信息
print("Loaded 3.x/4.x HD model.")
# 设置评估
model.eval()
# 绑定设备
model.device()

# 判断视频输入
if not args.video is None:
    # 打开视频
    videoCapture = cv2.VideoCapture(args.video)
    # 读取 fps
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    # 读取帧数
    tot_frame = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
    # 释放句柄
    videoCapture.release()
    # 判断是否自定 fps
    if args.fps is None:
        # 标记自动 fps
        fpsNotAssigned = True
        # 计算输出 fps
        args.fps = fps * args.multi
    else:
        # 标记自定 fps
        fpsNotAssigned = False
    # 创建帧生成器
    videogen = skvideo.io.vreader(args.video)
    # 读取首帧
    lastframe = next(videogen)
    # 定义编码
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    # 分离路径
    video_path_wo_ext, ext = os.path.splitext(args.video)
    # 打印信息
    print('{}.{}, {} frames in total, {}FPS to {}FPS'.format(video_path_wo_ext, args.ext, tot_frame, fps, args.fps))
    # 自动合成音频判断
    if args.png == False and fpsNotAssigned == True:
        # 打印合并提示
        print("The audio will be merged after interpolation process")
    else:
        # 打印不合并提示
        print("Will not merge audio because using png or fps flag!")
else:
    # 初始化列表
    videogen = []
    # 遍历目录
    for f in os.listdir(args.img):
        # 筛选 png
        if 'png' in f:
            # 添加文件
            videogen.append(f)
    # 统计帧数
    tot_frame = len(videogen)
    # 排序文件
    videogen.sort(key= lambda x:int(x[:-4]))
    # 读取首帧
    lastframe = cv2.imread(os.path.join(args.img, videogen[0]), cv2.IMREAD_UNCHANGED)[:, :, ::-1].copy()
    # 切片剩余
    videogen = videogen[1:]
# 读取尺寸
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
    # 判断自定义输出
    if args.output is not None:
        # 使用自定义
        vid_out_name = args.output
    else:
        # 生成默认名
        vid_out_name = '{}_{}X_{}fps.{}'.format(video_path_wo_ext, args.multi, int(np.round(args.fps)), args.ext)
    # 创建写入器
    vid_out = cv2.VideoWriter(vid_out_name, fourcc, args.fps, (w, h))

def clear_write_buffer(user_args, write_buffer):
    # 初始化计数
    cnt = 0
    # 循环消费
    while True:
        # 取出元素
        item = write_buffer.get()
        # 判断结束
        if item is None:
            # 跳出循环
            break
        # 判断 png 输出
        if user_args.png:
            # 写 PNG 文件
            cv2.imwrite('vid_out/{:0>7d}.png'.format(cnt), item[:, :, ::-1])
            # 计数加一
            cnt += 1
        else:
            # 写视频帧
            vid_out.write(item[:, :, ::-1])

def build_read_buffer(user_args, read_buffer, videogen):
    # 捕获异常
    try:
        # 遍历帧
        for frame in videogen:
            # 判断图片模式
            if not user_args.img is None:
                # 读取图片帧
                frame = cv2.imread(os.path.join(user_args.img, frame), cv2.IMREAD_UNCHANGED)[:, :, ::-1].copy()
            # 判断拼接模式
            if user_args.montage:
                # 裁剪居中
                frame = frame[:, left: left + w]
            # 放入队列
            read_buffer.put(frame)
    except:
        # 忽略异常
        pass
    # 发送结束
    read_buffer.put(None)

def make_inference(I0, I1, n):
    # 使用全局模型
    global model
    # 判断版本
    if model.version >= 3.9:
        # 初始化列表
        res = []
        # 遍历步数
        for i in range(n):
            # 推理时间点
            res.append(model.inference(I0, I1, (i+1) * 1. / (n+1), args.scale))
        # 返回结果
        return res
    else:
        # 推理中间帧
        middle = model.inference(I0, I1, args.scale)
        # 判断基线
        if n == 1:
            # 返回列表
            return [middle]
        # 递归左半
        first_half = make_inference(I0, middle, n=n//2)
        # 递归右半
        second_half = make_inference(middle, I1, n=n//2)
        # 奇偶分支
        if n%2:
            # 拼接包含中点
            return [*first_half, middle, *second_half]
        else:
            # 拼接不含中点
            return [*first_half, *second_half]

def pad_image(img):
    # 判断半精度
    if(args.fp16):
        # 返回半精度 pad
        return F.pad(img, padding).half()
    else:
        # 返回普通 pad
        return F.pad(img, padding)

# 判断拼接模式
if args.montage:
    # 计算左偏移
    left = w // 4
    # 更新宽度
    w = w // 2
# 计算基块
tmp = max(128, int(128 / args.scale))
# 计算 pad 高
ph = ((h - 1) // tmp + 1) * tmp
# 计算 pad 宽
pw = ((w - 1) // tmp + 1) * tmp
# 生成 padding
padding = (0, pw - w, 0, ph - h)
# 创建进度条
pbar = tqdm(total=tot_frame)
# 再次裁剪
if args.montage:
    # 裁剪 lastframe
    lastframe = lastframe[:, left: left + w]
# 创建写队列
write_buffer = Queue(maxsize=500)
# 创建读队列
read_buffer = Queue(maxsize=500)
# 启动读线程
_thread.start_new_thread(build_read_buffer, (args, read_buffer, videogen))
# 启动写线程
_thread.start_new_thread(clear_write_buffer, (args, write_buffer))

# 构造首张张量
I1 = torch.from_numpy(np.transpose(lastframe, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
# 填充首张
I1 = pad_image(I1)
# 初始化临时帧
temp = None

# 循环读取
while True:
    # 判断缓存帧
    if temp is not None:
        # 使用缓存
        frame = temp
        # 清空缓存
        temp = None
    else:
        # 读取新帧
        frame = read_buffer.get()
    # 结束判断
    if frame is None:
        # 跳出循环
        break
    # 更新上一帧
    I0 = I1
    # 生成当前帧张量
    I1 = torch.from_numpy(np.transpose(frame, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
    # 填充当前帧
    I1 = pad_image(I1)
    # 下采样上一帧
    I0_small = F.interpolate(I0, (32, 32), mode='bilinear', align_corners=False)
    # 下采样当前帧
    I1_small = F.interpolate(I1, (32, 32), mode='bilinear', align_corners=False)
    # 计算结构相似
    ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])
    # 初始化标志
    break_flag = False
    # 静态检测
    if ssim > 0.996:
        # 读取下一帧
        frame = read_buffer.get()
        # 判断结束
        if frame is None:
            # 设置标志
            break_flag = True
            # 使用 lastframe
            frame = lastframe
        else:
            # 缓存帧
            temp = frame
        # 重新生成张量
        I1 = torch.from_numpy(np.transpose(frame, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
        # 填充张量
        I1 = pad_image(I1)
        # 模型推理
        I1 = model.inference(I0, I1, scale=args.scale)
        # 下采样新帧
        I1_small = F.interpolate(I1, (32, 32), mode='bilinear', align_corners=False)
        # 重新计算 ssim
        ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])
        # 转换为图像
        frame = (I1[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]
    # 判断低相似
    if ssim < 0.2:
        # 初始化输出
        output = []
        # 填充重复帧
        for i in range(args.multi - 1):
            # 添加 I0
            output.append(I0)
    else:
        # 正常插帧
        output = make_inference(I0, I1, args.multi - 1)
    # 判断拼接输出
    if args.montage:
        # 写入原帧拼接
        write_buffer.put(np.concatenate((lastframe, lastframe), 1))
        # 遍历中间帧
        for mid in output:
            # 转换中间帧
            mid = (((mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)))
            # 写入拼接
            write_buffer.put(np.concatenate((lastframe, mid[:h, :w]), 1))
    else:
        # 写入上一帧
        write_buffer.put(lastframe)
        # 遍历中间帧
        for mid in output:
            # 转换张量
            mid = (((mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)))
            # 写入帧
            write_buffer.put(mid[:h, :w])
    # 更新进度
    pbar.update(1)
    # 更新 lastframe
    lastframe = frame
    # 判断结束标志
    if break_flag:
        # 跳出循环
        break

# 拼接模式输出尾帧
if args.montage:
    # 写入拼接尾帧
    write_buffer.put(np.concatenate((lastframe, lastframe), 1))
else:
    # 写入尾帧
    write_buffer.put(lastframe)
# 写入结束标记
write_buffer.put(None)

# 导入时间模块
import time
# 等待队列清空
while(not write_buffer.empty()):
    # 小睡片刻
    time.sleep(0.1)
# 关闭进度条
pbar.close()
# 释放 writer
if not vid_out is None:
    # 释放资源
    vid_out.release()

# 判断是否复制音频
if args.png == False and fpsNotAssigned == True and not args.video is None:
    # 尝试复制
    try:
        # 调用复制函数
        transferAudio(args.video, vid_out_name)
    except:
        # 捕获失败
        print("Audio transfer failed. Interpolated video will have no audio")
        # 生成无音频名
        targetNoAudio = os.path.splitext(vid_out_name)[0] + "_noaudio" + os.path.splitext(vid_out_name)[1]
        # 回退重命名
        os.rename(targetNoAudio, vid_out_name)
