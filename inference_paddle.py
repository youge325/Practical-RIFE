import os
import cv2
import paddle
import numpy as np
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

# 设置设备
paddle.set_device('gpu' if paddle.is_compiled_with_cuda() else 'cpu')
paddle.set_grad_enabled(False)

# 导入模型
from train_log.RIFE_HDv3_paddle import Model
import pickle

# 实例化模型
model = Model()
# 加载权重
with open('train_log/flownet_paddle.pdparams', 'rb') as f:
    state_dict = pickle.load(f)
model.flownet.set_dict(state_dict)
model.eval()
print("Model loaded successfully")

# 读取图像
def read_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, (448, 256))
    img = img.astype(np.float32) / 255.0
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=2)  # 如果是灰度，转 RGB
    elif img.shape[2] == 4:
        img = img[:, :, :3]  # 移除 alpha
    img = np.transpose(img, (2, 0, 1))  # HWC to CHW
    img = np.expand_dims(img, 0)  # add batch dim
    return paddle.to_tensor(img)

# 推理函数
def inference(img0, img1, exp=4):
    middle = model.inference(img0, img1, exp)
    return middle

# 主函数
if __name__ == "__main__":
    img0_path = 'demo/i0.png'
    img1_path = 'demo/i1.png'
    exp = 4  # 插帧倍数

    img0 = read_image(img0_path)
    img1 = read_image(img1_path)

    print("Starting inference...")
    output = inference(img0, img1, exp)
    print("Inference completed")

    # 保存输出
    output = output.numpy()
    output = np.squeeze(output, 0)  # remove batch dim
    output = np.transpose(output, (1, 2, 0))  # CHW to HWC
    output = (output * 255.0).astype(np.uint8)
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    cv2.imwrite('output.png', output)
    print("Output saved to output.png")