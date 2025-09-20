import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
import torch.optim as optim
import itertools
from model.warplayer import warp
from torch.nn.parallel import DistributedDataParallel as DDP
from train_log.IFNet_HDv3 import *
import torch.nn.functional as F
from model.loss import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
class Model:
    # 高分辨率插帧模型封装：包含光流网络、优化器、损失组件
    def __init__(self, local_rank=-1):
        self.flownet = IFNet()  # 主干插帧光流与融合网络
        self.device()           # 迁移到计算设备
        self.optimG = AdamW(self.flownet.parameters(), lr=1e-6, weight_decay=1e-4)  # 优化器
        self.epe = EPE()        # 光流端点误差（若训练时使用）
        self.version = 4.25     # 模型版本号（外部逻辑依据）
        # self.vgg = VGGPerceptualLoss().to(device)  # 可选感知损失
        self.sobel = SOBEL()    # 光流平滑/边缘约束
        if local_rank != -1:    # 分布式训练包装
            self.flownet = DDP(self.flownet, device_ids=[local_rank], output_device=local_rank)

    def train(self):
        # 切换到训练模式（启用 BN/Dropout 等）
        self.flownet.train()

    def eval(self):
        # 切换到推理模式
        self.flownet.eval()

    def device(self):
        # 将网络放置到预设设备（GPU/CPU）
        self.flownet.to(device)

    def load_model(self, path, rank=0):
        # 载入模型权重，兼容 DDP 保存的 'module.' 前缀
        def convert(param):
            if rank == -1:
                return {
                    k.replace("module.", ""): v
                    for k, v in param.items()
                    if "module." in k
                }
            else:
                return param
        if rank <= 0:
            if torch.cuda.is_available():
                self.flownet.load_state_dict(convert(torch.load('{}/flownet.pkl'.format(path))), False)
            else:
                self.flownet.load_state_dict(convert(torch.load('{}/flownet.pkl'.format(path), map_location ='cpu')), False)
        
    def save_model(self, path, rank=0):
        # 保存当前光流网络权重
        if rank == 0:
            torch.save(self.flownet.state_dict(),'{}/flownet.pkl'.format(path))

    def inference(self, img0, img1, timestep=0.5, scale=1.0):
        # 前向推理：输入两帧与目标时间（timestep 可为标量或张量）
        imgs = torch.cat((img0, img1), 1)
        # 不同层级的尺度控制列表（随 scale 调整）
        scale_list = [16/scale, 8/scale, 4/scale, 2/scale, 1/scale]
        flow, mask, merged = self.flownet(imgs, timestep, scale_list)
        # 返回最终融合结果（列表最后一个元素）
        return merged[-1]
    
    def update(self, imgs, gt, learning_rate=0, mul=1, training=True, flow_gt=None):
        # 单次优化 / 前向更新：返回预测结果与损失字典
        for param_group in self.optimG.param_groups:
            param_group['lr'] = learning_rate
        img0 = imgs[:, :3]
        img1 = imgs[:, 3:]
        if training:
            self.train()
        else:
            self.eval()
        scale = [16, 8, 4, 2, 1]
        flow, mask, merged = self.flownet(torch.cat((imgs, gt), 1), scale=scale, training=training)
        loss_l1 = (merged[-1] - gt).abs().mean()  # 重建 L1 损失
        loss_smooth = self.sobel(flow[-1], flow[-1]*0).mean()  # 光流平滑（与零梯度比较）
        # loss_vgg = self.vgg(merged[-1], gt)
        # 占位一致性损失（原实现可能来自附加模块，这里置 0 以保持训练兼容）
        loss_cons = torch.tensor(0.0, device=gt.device)
        if training:
            self.optimG.zero_grad()
            loss_G = loss_l1 + loss_cons + loss_smooth * 0.1
            loss_G.backward()
            self.optimG.step()
        else:
            flow_teacher = flow[2]
        return merged[-1], {
            'mask': mask,
            'flow': flow[-1][:, :2],
            'loss_l1': loss_l1,
            'loss_cons': loss_cons,
            'loss_smooth': loss_smooth,
            }
