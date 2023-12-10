# 数据集部分

https://www.dropbox.com/s/10bfat0kg4si1bu/zalando-hd-resized.zip?dl=0

## 原始数据集结构

    data
        train_pairs.txt
        test_pairs.txt
        train
            agnostic-v3.2
            cloth
            cloth-mask
            image
            image-densepose
            image-parse-agnostic-v3.2
            image-parse-v3
            openpose_img
            openpose_json

        test
            agnostic-v3.2
            cloth
            cloth-mask
            image
            image-densepose
            image-parse-agnostic-v3.2
            image-parse-v3
            openpose_img
            openpose_json


## 预处理后的结构

    data
        train:10080
            ip # 目标人的图片目录
            ig # 目标衣服的图片目录
            ia # ip处理后无衣服的人图目录
            ic # ig处理后无人的衣服图目录
            jp # 通过ip处理后得到的 Person pose 目录
            jg # 通过ig处理后得到的 Garment pose 目录
            ig_mask # 分割衣服的mask图目录
            itr128 # unet128网络训练结果存放目录
            itr128_ema # unet128网络+ema训练结果存放目录

        val:1904
            ip # 目标人的图片目录
            ig # 目标衣服的图片目录
            ia # ip处理后无衣服的人图目录
            ic # ig处理后无人的衣服图目录
            jp # 通过ip处理后得到的 Person pose 目录
            jg # 通过ig处理后得到的 Garment pose 目录
            ig_mask # 分割衣服的mask图目录
            itr128 # unet128网络训练结果存放目录
            itr128_ema # unet128网络+ema训练结果存放目录

        test
            ip # 目标人的图片目录
            ig # 目标衣服的图片目录

        pose_output # mmpose生成的姿态识别结果（json格式）
            train:10080
                ip
                    predictions
                    visualizations
                ig
                    predictions
                    visualizations
            val:1904
                ip
                    predictions
                    visualizations
                ig
                    predictions
                    visualizations
            test

