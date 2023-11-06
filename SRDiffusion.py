# -*- coding: utf-8 -*-
# @Author        : wenzonghui
# @Email         : wenzonghui@gmail.com
# @Create Date   : 2023-11-05 22:44:55
# @Last Modified : 2023-11-05 22:44:55
# @Description   :

import torch
from torch import nn

from network_elem import DownSample, UpSample, SRBlock


# 这个SR超分辨率模块还没有验证，可采用 imagen 来实现超分辨率的效果
class SRDiffusion(nn.Module):

    def __init__(self, device, time_dim=512):
        super().__init__()

        # initial image embedding size 256x256 and 6 channels(concatenate rgb agnostic image and noise)

        # 3x3 conv layer person
        # 输入通道 3: 因为图片通道是 3
        self.init_conv_person = nn.Conv2d(3, 128, (3, 3), padding=1)

        # 256 unit encoder person
        self.block1_person = SRBlock(block_channel=128, res_blocks_number=2)
        self.downsample1_person = SRDownSample(dim=128, dim_out=128, t_emb_dim=time_dim)

        # 128 unit encoder person
        self.block2_person = SRBlock(block_channel=128, res_blocks_number=3)
        self.downsample2_person = SRDownSample(dim=128, dim_out=256, t_emb_dim=time_dim)

        # 64 unit encoder person
        self.block3_person = SRBlock(block_channel=256, res_blocks_number=4)
        self.downsample3_person = SRDownSample(dim=256, dim_out=512, t_emb_dim=time_dim)

        # 32 unit encoder person
        self.block4_person = SRBlock(block_channel=512, res_blocks_number=7)
        self.downsample4_person = SRDownSample(dim=512, dim_out=1024, t_emb_dim=time_dim)

        # 16 unit encoder person
        self.block5_person = SRBlock(block_channel=1024, res_blocks_number=7)

        # 16 unit decoder person
        self.block6_person = SRBlock(block_channel=1024, res_blocks_number=7)
        self.upsample6_person = SRUpSample(dim=2048, dim_out=512, t_emb_dim=time_dim)

        # 32 unit decoder person
        self.block7_person = SRBlock(block_channel=512, res_blocks_number=7)
        self.upsample7_person = SRUpSample(dim=1024, dim_out=256, t_emb_dim=time_dim)

        # 64 unit decoder person
        self.block8_person = SRBlock(block_channel=256, res_blocks_number=4)
        self.upsample8_person = SRUpSample(dim=512, dim_out=128, t_emb_dim=time_dim)

        # 128 unit decoder person
        self.block9_person = SRBlock(block_channel=128, res_blocks_number=3)
        self.upsample9_person = SRUpSample(dim=256, dim_out=128, t_emb_dim=time_dim)

        # 256 unit decoder person
        self.block10_person = SRBlock(block_channel=128, res_blocks_number=2)

        self.final_conv_person = nn.Conv2d(256, 3, (3, 3), padding=1)

    def forward(self, zt, ic, time_step, noise_level):
        """

        :param zt: rgb_agnostic and noisy ground truth concatenated across channels
        :param ic: segmented garment
        :param time_step:
        :param noise_level: sigma sampled from uniform distribution
        :return:
        """

        pos_encoding = self.pos_encod_layer(time_step)

        # passing through initial conv layer: zt
        # out: 256 * 256 * 128
        zt_init = self.init_conv_person(zt)
        print('zt_init: ', zt_init.size())

        # 256 - person
        # out: 256 * 256 * 128
        zt_256_1 = self.block1_person(zt_init, clip_embedding)
        print('zt_256_1: ', zt_256_1.size())

        # 128 - person
        # out: 128 * 128 * 128
        zt_128_downsample = self.downsample1_person(zt_256_1, pos_encoding)
        zt_128_1 = self.block2_person(zt_128_downsample)
        print('zt_128_1: ', zt_128_1.size())

        # 64 - person
        # out: 64 * 64 * 256
        zt_64_1_downsample = self.downsample2_person(zt_128_1, pos_encoding)
        zt_64_1 = self.block3_person(zt_64_1_downsample)
        print('zt_64_1_downsample: ', zt_64_1_downsample.size())
        print('zt_64_1: ', zt_64_1.size())

        # 32 - person
        # out: 32 * 32 * 512
        zt_32_1_downsample = self.downsample3_person(zt_64_1, pos_encoding)
        zt_32_1 = self.block4_person(zt_32_1_downsample)
        print('zt_32_1_downsample: ', zt_32_1_downsample.size())
        print('zt_32_1: ', zt_32_1.size())

        # 16 - person
        # out: 16 * 16 * 1024
        zt_16_1_downsample = self.downsample4_person(zt_32_1, pos_encoding)
        zt_16_1 = self.block5_person(zt_16_1_downsample)
        print('zt_16_1_downsample: ', zt_16_1_downsample.size())
        print('zt_16_1: ', zt_16_1.size())

        # --------------------------------------------- 右边一半 ---------------------------------------------

        # 16 - person
        # out: 16 * 16 * 2048
        zt_16_2_noconcat = self.block6_person(zt_16_1)
        zt_16_2 = torch.concat((zt_16_2_noconcat, zt_16_1), dim=1)
        # zt_16_2 = zt_16_2 + zt_16_1
        print('zt_16_2_noconcat: ', zt_16_2_noconcat.size())
        print('zt_16_2: ', zt_16_2.size())

        # 32 - person
        # out: 32 * 32 * 1024
        zt_32_2_upsample = self.upsample6_person(zt_16_2, pos_encoding)
        zt_32_2_noconcat = self.block7_person(zt_32_2_upsample)
        zt_32_2 = torch.concat((zt_32_2_noconcat, zt_32_1), dim=1)
        print('zt_32_2_upsample: ', zt_32_2_upsample.size())
        print('zt_32_2_noconcat: ', zt_32_2_noconcat.size())
        print('zt_32_2: ', zt_32_2.size())

        # 64 - person
        # out: 64 * 64 * 512
        zt_64_2_upsample = self.upsample7_person(zt_32_2, pos_encoding)
        zt_64_2_noconcat = self.block8_person(zt_64_2_upsample)
        zt_64_2 = torch.concat((zt_64_2_noconcat, zt_64_1), dim=1)
        print('zt_64_2_upsample: ', zt_64_2_upsample.size())
        print('zt_64_2_noconcat: ', zt_64_2_noconcat.size())
        print('zt_64_2: ', zt_64_2.size())

        # 128 - person
        # out: 128 * 128 * 512
        zt_128_2_upsample = self.upsample8_person(zt_64_2, pos_encoding)
        zt_128_2_noconcat = self.block9_person(zt_128_2_upsample)
        zt_128_2 = torch.concat((zt_128_2_noconcat, zt_128_1), dim=1)
        print('zt_128_2_upsample: ', zt_128_2_upsample.size())
        print('zt_128_2_noconcat: ', zt_128_2_noconcat.size())
        print('zt_128_2: ', zt_128_2.size())

        # 256 - person
        # out: 256 * 256 * 256
        zt_256_2_upsample = self.upsample9_person(zt_128_2, pos_encoding)
        zt_256_2_noconcat = self.block10_person(zt_256_2_upsample)
        zt_256_2 = torch.concat((zt_256_2_noconcat, zt_256_1), dim=1)
        print('zt_256_2_noconcat: ', zt_256_2_noconcat.size())
        print('zt_256_2: ', zt_256_2.size())

        # final conv layer - person
        # out: 256 * 256 * 3
        zt_final = self.final_conv_person(zt_256_2)
        print('zt_final: ', zt_final.size())

        return zt_final


if __name__ == "__main__":
    time_step = torch.randint(low=1, high=1000, size=(4,))
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    # 测试 SRDiffusion 网络
    sr1024 = SRDiffusion(device).to(device)
    # def forward(self, zt, ic, person_pose_embedding, garment_pose_embedding, time_step, noise_level):
    # 输入通道 9 是因为三组图片 concat： Itr128、Ia、Zt，三个图片 concat 为 Zt
    out1024 = sr1024(torch.randn(4, 3, 256, 256).to(device),
                    torch.randn(4, 8).to(device),
                    torch.randn(4, 8).to(device),
                    time_step.to(device),
                    0.3)

    print("out1024: ", out1024.size())