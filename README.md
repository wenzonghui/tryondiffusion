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

## tensorboard

```
pip install --upgrade pip
pip install nvidia-pyindex
pip install nvidia-tensorflow[horovod]
pip install nvidia-tensorboard==1.15
```

## 

## 本项目

```
pip install -r requirements.txt
```