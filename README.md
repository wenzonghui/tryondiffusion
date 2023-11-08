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

"Given an image Ip of person p and image Ig of a different person in garment g"

预处理脚本在 pre_processing 文件夹。