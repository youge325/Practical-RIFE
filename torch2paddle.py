import numpy as np
import torch
import pickle
from collections import OrderedDict

def torch2paddle():
    torch_path = "./train_log/flownet.pkl"
    paddle_path = "./train_log/flownet_paddle.pdparams"
    torch_state_dict = torch.load(torch_path)
    fc_names = []  # 如果有 Linear 层，添加名称
    paddle_state_dict = OrderedDict()
    
    for k in torch_state_dict:
        if "num_batches_tracked" in k:  # 飞桨中无此参数，无需保存
            continue
        v = torch_state_dict[k].detach().cpu().numpy()
        flag = [i in k for i in fc_names]
        if any(flag) and "weight" in k:
            new_shape = [1, 0] + list(range(2, v.ndim))
            print(
                f"name: {k}, ori shape: {v.shape}, new shape: {v.transpose(new_shape).shape}"
            )
            v = v.transpose(new_shape)   # 转置 Linear 层的 weight 参数
        # 将 torch.nn.BatchNorm2d 的参数名称改成 paddle.nn.BatchNorm2D 对应的参数名称
        k = k.replace("running_var", "_variance")
        k = k.replace("running_mean", "_mean")
        # 移除 'module.' 前缀，因为 PaddlePaddle 不使用 DataParallel 前缀
        k = k.replace("module.", "")
        # 添加到飞桨权重字典中
        paddle_state_dict[k] = v
    with open(paddle_path, 'wb') as f:
        pickle.dump(paddle_state_dict, f)
    print(f"Converted weights saved to {paddle_path}")

if __name__ == "__main__":
    torch2paddle()