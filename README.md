# tryondiffusion

## 论文

https://tryondiffusion.github.io/

https://arxiv.org/abs/2306.08276

## 已知开源代码

https://github.com/Mutoy-choi/Tryondiffusion

https://github.com/kailashahirwar/tryondiffusion/tree/main

## 核心组件

分为3个部分，需要分开训练3个模型；

- UNet128
- UNet256
- SRDiffusion

UNet128 与 UNet256 已经验证完毕，SRDiffusion 超分辨率部分采用 imagen 来做。

https://github.com/lucidrains/imagen-pytorch

## 数据预处理

预处理脚本在 pre_processing 文件夹

数据集处理：包括数据集两两配对，打乱
从 json 生成 mask 图
由原图扣出衣服图
由原图扣出人


## 安装

### pytorch

选择合适的版本
```
https://pytorch.org/get-started/locally/
```

### segment anything

进入虚拟环境，运行 
```
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install opencv-python pycocotools matplotlib onnxruntime onnx
```

### mmpose

```
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.1"
mim install "mmdet>=3.1.0"
cd mmpose
pip install -r requirements.txt
pip install -v -e .
```

### tensorboard
默认安装最新版
```
pip install tensorboard
```
或者安装版本`1.15`
```
pip install --upgrade pip
pip install nvidia-pyindex
pip install nvidia-tensorflow[horovod]
pip install nvidia-tensorboard==1.15
```

### 本项目

```
pip install -r requirements.txt
```

## 训练

### 前期准备
`cd`进入本项目目录下
`screen`进入虚拟窗口开始训练再推出窗口，防止`ssh`断连导致训练失败
```
pip install screen  # 安装虚拟窗口 screen
screen -S train  # 创建虚拟窗口 train
screen -ls  # 列出所有虚拟窗口
screen -r train  # 进入虚拟窗口 train
Ctrl + A + D  # 隐藏虚拟窗口
exit  # 删除虚拟窗口
```

### 普通单卡模式运行
```
修改 trainer.py 内对应的文件路径
python trainer.py
```

### 单机多卡 DP 模式运行
```
修改 trainer_DP.py 内对应的文件路径
python trainer_DP.py
```

### 单机多卡 DDP 模式运行
```
修改 trainer_DDP.py 内对应的文件路径
python -m torch.distributed.launch --nproc_per_node=4 trainer_DDP.py --use_mix_precision -b=8 -e=200 -d=128 --logdir=runs/unet128
```

### 多机多卡 DDP 模式运行
```
修改 trainer_DDP.py 内对应的文件路径

# 两台电脑，每台8卡情况下的启动方式
# nproc_per_node: 每台机器中运行几个进程
# nnodes：一共使用多少台机器
# node_rank：当前机器的序号
# master_addr：0号机器的IP
# master_port：0号机器的可用端口

# Node 0
# ip 192.168.1.202  port : 29501
# terminal-0
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr="192.168.1.202" --master_port=29501 trainer_DDP.py --use_mix_precision -b=2 -e=200 -d=128 --logdir=runs/unet128

# Node 1
# terminal-0
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr="192.168.1.202" --master_port=29501 trainer_DDP.py --use_mix_precision -b=2 -e=200 -d=128 --logdir=runs/unet128
```

## 推理
推理模式务必要符合相应的训练模式

### 普通单卡模式推理


### 单机多卡 DP 模式推理


### 单机多卡 DDP 模式推理


### 多机多卡 DDP 模式推理
