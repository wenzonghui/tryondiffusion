# 训练前期准备

## Jp: Person pose embedding
Person pose --> FC --> Person pose embedding

## Jg: Garment pose embedding
Garment pose --> FC --> Garment pose embedding

## Ia: Clothing agnostic RGB
Ip --> Ia

## Ic: Segmented garment
Ig --> Ic

## Zt: Noisy image
Ip --> Zt
Zt concat Ia --> Zt


## 这一步承接上一步mmpose得到的姿势估计，通过pose_garment_jsonpreprocess.py处理得到如下数据

    pose_output
        train
            ip_pose_json
            ig_pose_json
        val
            ip_pose_json
            ig_pose_json

## 然后，分别通过person_pose_embedding和garment_pose_embedding进行fc层网络训练

分别运行 train.py 得到两个 embed model