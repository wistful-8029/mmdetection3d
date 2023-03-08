# 读取配置文件
from mmcv import Config

config_file = "/home/wistful/work/mmdetection3d/configs/my_config/my_config.py"
cfg = Config.fromfile(config_file)
print("cfg type:", type(cfg))
print("cfg.model type:", type(cfg.model))
print(cfg.data.train.get('modality'))

import os

os.chdir('/home/wistful/work/mmdetection3d/')
from torchvision.transforms import transforms  # 取数据
from mmdet3d.datasets import build_dataset
import matplotlib.pyplot as plt

# 读取数据集
datasets = [build_dataset(cfg.data.train)]

print(len(datasets))
# type(datasets[0])
print(datasets[0][0].keys())
