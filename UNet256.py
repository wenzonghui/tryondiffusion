# -*- coding: utf-8 -*-
# @Author        : wenzonghui
# @Email         : wenzonghui@gmail.com
# @Create Date   : 2023-11-04 22:23:28
# @Last Modified : 2023-11-04 22:23:28
# @Description   :

import torch
from torch import nn

from network_elem import UNetBlockNoAttention, DownSample, UNetBlockAttention, UpSample, SinusoidalPosEmbed, \
    AttentionPool1d


class UNet256(nn.Module):

    def __init__(self, pose_embed_len_dim, device, time_dim=512):
        super().__init__()

        self.pos_encod_layer = SinusoidalPosEmbed(time_dim)

        # process clip embeddings
        self.attn_pool_layer = AttentionPool1d(pose_embed_len_dim, device)

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> person UNet >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # initial image embedding size 256x256 and 6 channels(concatenate rgb agnostic image and noise)

        # 3x3 conv layer person
        # 输入通道 9 是因为三组图片 concat： Itr128、Ia、Zt，三个图片 concat 为 Zt
        self.init_conv_person = nn.Conv2d(9, 128, (3, 3), padding=1)

        # 256 unit encoder person
        self.block1_person = UNetBlockNoAttention(block_channel=128, clip_dim=pose_embed_len_dim, res_blocks_number=2)
        self.downsample1_person = DownSample(dim=128, dim_out=128, t_emb_dim=time_dim)

        # 128 unit encoder person
        self.block2_person = UNetBlockNoAttention(block_channel=128, clip_dim=pose_embed_len_dim, res_blocks_number=3)
        self.downsample2_person = DownSample(dim=128, dim_out=256, t_emb_dim=time_dim)

        # 64 unit encoder person
        self.block3_person = UNetBlockNoAttention(block_channel=256, clip_dim=pose_embed_len_dim, res_blocks_number=4)
        self.downsample3_person = DownSample(dim=256, dim_out=512, t_emb_dim=time_dim)

        # 32 unit encoder person
        self.block4_person = UNetBlockNoAttention(block_channel=512, clip_dim=pose_embed_len_dim, res_blocks_number=7)
        self.downsample4_person = DownSample(dim=512, dim_out=1024, t_emb_dim=time_dim)

        # 16 unit encoder person
        self.block5_person = UNetBlockAttention(block_channel=1024, clip_dim=pose_embed_len_dim, res_blocks_number=7, hw_dim=16)

        # 16 unit decoder person
        self.block6_person = UNetBlockAttention(block_channel=1024, clip_dim=pose_embed_len_dim, res_blocks_number=7, hw_dim=16)
        self.upsample6_person = UpSample(dim=2048, dim_out=512, t_emb_dim=time_dim)

        # 32 unit decoder person
        self.block7_person = UNetBlockNoAttention(block_channel=512, clip_dim=pose_embed_len_dim, res_blocks_number=7)
        self.upsample7_person = UpSample(dim=1024, dim_out=256, t_emb_dim=time_dim)

        # 64 unit decoder person
        self.block8_person = UNetBlockNoAttention(block_channel=256, clip_dim=pose_embed_len_dim, res_blocks_number=4)
        self.upsample8_person = UpSample(dim=512, dim_out=128, t_emb_dim=time_dim)

        # 128 unit decoder person
        self.block9_person = UNetBlockNoAttention(block_channel=128, clip_dim=pose_embed_len_dim, res_blocks_number=3)
        self.upsample9_person = UpSample(dim=256, dim_out=128, t_emb_dim=time_dim)

        # 256 unit decoder person
        self.block10_person = UNetBlockNoAttention(block_channel=128, clip_dim=pose_embed_len_dim, res_blocks_number=2)

        self.final_conv_person = nn.Conv2d(256, 3, (3, 3), padding=1)

        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< person UNet ends <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> garment UNet >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # initial image embedding size 128x128 and 3 channels

        # 3x3 conv layer garment
        self.init_conv_garment = nn.Conv2d(3, 128, (3, 3), padding=1)

        # 256 unit encoder garment
        self.block1_garment = UNetBlockNoAttention(block_channel=128, clip_dim=pose_embed_len_dim, res_blocks_number=2)
        self.downsample1_garment = DownSample(dim=128, dim_out=125, t_emb_dim=time_dim)

        # 128 unit encoder garment
        self.block2_garment = UNetBlockNoAttention(block_channel=128, clip_dim=pose_embed_len_dim, res_blocks_number=3)
        self.downsample2_garment = DownSample(dim=128, dim_out=256, t_emb_dim=time_dim)

        # 64 unit encoder garment
        self.block3_garment = UNetBlockNoAttention(block_channel=256, clip_dim=pose_embed_len_dim, res_blocks_number=4)
        self.downsample3_garment = DownSample(dim=256, dim_out=512, t_emb_dim=time_dim)

        # 32 unit encoder garment
        self.block4_garment = UNetBlockNoAttention(block_channel=512, clip_dim=pose_embed_len_dim, res_blocks_number=6)
        self.downsample4_garment = DownSample(dim=512, dim_out=1024, t_emb_dim=time_dim)

        # 16 unit encoder garment
        self.block5_garment = UNetBlockNoAttention(block_channel=1024, clip_dim=pose_embed_len_dim, res_blocks_number=7)

        # 16 unit decoder garment
        self.block6_garment = UNetBlockNoAttention(block_channel=1024, clip_dim=pose_embed_len_dim, res_blocks_number=7)

        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< garment UNet ends <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


    def forward(self, zt, ic, person_pose_embedding, garment_pose_embedding, time_step, noise_level):
        """

        :param zt: rgb_agnostic and noisy ground truth concatenated across channels
        :param ic: segmented garment
        :param person_pose_embedding: [b, 1, vector_length]
        :param garment_pose_embedding: [b, 1, vector_length]
        :param time_step:
        :param noise_level: sigma sampled from uniform distribution
        :return:
        """

        pos_encoding = self.pos_encod_layer(time_step)

        # concat pose embeddings
        pose_embeddings = torch.concat((person_pose_embedding[:, None, :], garment_pose_embedding[:, None, :]), dim=1)

        # get clip embeddings
        clip_embedding = self.attn_pool_layer(pose_embeddings, time_step, noise_level)

        # passing through initial conv layer: ic
        # out: 256 * 256 * 128
        ic_init = self.init_conv_garment(ic)
        print('ic_init: ', ic_init.size())

        # passing through initial conv layer: zt
        # out: 256 * 256 * 128
        zt_init = self.init_conv_person(zt)
        print('zt_init: ', zt_init.size())

        # entering UNet
        # 256 - garment
        # out: 256 * 256 * 128
        ic_256 = self.block1_garment(ic_init, clip_embedding)
        print('ic_256: ', ic_256.size())

        # 256 - person
        # out: 256 * 256 * 128
        zt_256_1 = self.block1_person(zt_init, clip_embedding)
        print('zt_256_1: ', zt_256_1.size())

        # 128 - garment
        # out: 128 * 128 * 128
        ic_128_downsample = self.downsample1_garment(ic_256, pos_encoding)
        ic_128 = self.block2_garment(ic_128_downsample, clip_embedding)
        print('ic_128: ', ic_128.size())

        # 128 - person
        # out: 128 * 128 * 128
        zt_128_downsample = self.downsample1_person(zt_256_1, pos_encoding)
        zt_128_1 = self.block2_person(zt_128_downsample, clip_embedding)
        print('zt_128_1: ', zt_128_1.size())

        # 64 - garment
        # out: 64 * 64 * 256
        ic_64_downsample = self.downsample2_garment(ic_128, pos_encoding)
        ic_64 = self.block3_garment(ic_64_downsample, clip_embedding)
        print('ic_64_downsample: ', ic_64_downsample.size())
        print('ic_64: ', ic_64.size())

        # 64 - person
        # out: 64 * 64 * 256
        zt_64_1_downsample = self.downsample2_person(zt_128_1, pos_encoding)
        zt_64_1 = self.block3_person(zt_64_1_downsample, clip_embedding)
        print('zt_64_1_downsample: ', zt_64_1_downsample.size())
        print('zt_64_1: ', zt_64_1.size())

        # 32 - garment
        # out: 32 * 32 * 512
        ic_32_downsample = self.downsample3_garment(ic_64, pos_encoding)
        ic_32 = self.block4_garment(ic_32_downsample, clip_embedding)
        print('ic_32_downsample: ', ic_32_downsample.size())
        print('ic_32: ', ic_32.size())

        # 32 - person
        # out: 32 * 32 * 512
        zt_32_1_downsample = self.downsample3_person(zt_64_1, pos_encoding)
        zt_32_1 = self.block4_person(zt_32_1_downsample, clip_embedding)
        print('zt_32_1_downsample: ', zt_32_1_downsample.size())
        print('zt_32_1: ', zt_32_1.size())

        # 16 - garment
        # out: 16 * 16 * 1024
        ic_16_1_downsample = self.downsample4_garment(ic_32, pos_encoding)
        ic_16_1 = self.block5_garment(ic_16_1_downsample, clip_embedding)
        print('ic_16_1_downsample: ', ic_16_1_downsample.size())
        print('ic_16_1: ', ic_16_1.size())

        # 16 - person
        # out: 16 * 16 * 1024
        zt_16_1_downsample = self.downsample4_person(zt_32_1, pos_encoding)
        zt_16_1 = self.block5_person(zt_16_1_downsample, clip_embedding, pose_embeddings, ic_16_1)
        print('zt_16_1_downsample: ', zt_16_1_downsample.size())
        print('zt_16_1: ', zt_16_1.size())

        # --------------------------------------------- 右边一半 ---------------------------------------------
        # 16 - garment
        # out: 16 * 16 * 2048
        ic_16_2_noconcat = self.block6_garment(ic_16_1, clip_embedding)
        ic_16_2 = torch.concat((ic_16_2_noconcat, ic_16_1), dim=1)
        print('ic_16_2_noconcat: ', ic_16_2_noconcat.size())
        print('ic_16_2: ', ic_16_2.size())

        # 16 - person
        # out: 16 * 16 * 2048
        zt_16_2_noconcat = self.block6_person(zt_16_1, clip_embedding, pose_embeddings, ic_16_2)
        zt_16_2 = torch.concat((zt_16_2_noconcat, zt_16_1), dim=1)
        # zt_16_2 = zt_16_2 + zt_16_1
        print('zt_16_2_noconcat: ', zt_16_2_noconcat.size())
        print('zt_16_2: ', zt_16_2.size())

        # 32 - person
        # out: 32 * 32 * 1024
        zt_32_2_upsample = self.upsample6_person(zt_16_2, pos_encoding)
        zt_32_2_noconcat = self.block7_person(zt_32_2_upsample, clip_embedding)
        zt_32_2 = torch.concat((zt_32_2_noconcat, zt_32_1), dim=1)
        print('zt_32_2_upsample: ', zt_32_2_upsample.size())
        print('zt_32_2_noconcat: ', zt_32_2_noconcat.size())
        print('zt_32_2: ', zt_32_2.size())

        # 64 - person
        # out: 64 * 64 * 512
        zt_64_2_upsample = self.upsample7_person(zt_32_2, pos_encoding)
        zt_64_2_noconcat = self.block8_person(zt_64_2_upsample, clip_embedding)
        zt_64_2 = torch.concat((zt_64_2_noconcat, zt_64_1), dim=1)
        print('zt_64_2_upsample: ', zt_64_2_upsample.size())
        print('zt_64_2_noconcat: ', zt_64_2_noconcat.size())
        print('zt_64_2: ', zt_64_2.size())

        # 128 - person
        # out: 128 * 128 * 512
        zt_128_2_upsample = self.upsample8_person(zt_64_2, pos_encoding)
        zt_128_2_noconcat = self.block9_person(zt_128_2_upsample, clip_embedding)
        zt_128_2 = torch.concat((zt_128_2_noconcat, zt_128_1), dim=1)
        print('zt_128_2_upsample: ', zt_128_2_upsample.size())
        print('zt_128_2_noconcat: ', zt_128_2_noconcat.size())
        print('zt_128_2: ', zt_128_2.size())

        # 256 - person
        # out: 256 * 256 * 256
        zt_256_2_upsample = self.upsample9_person(zt_128_2, pos_encoding)
        zt_256_2_noconcat = self.block10_person(zt_256_2_upsample, clip_embedding)
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

    # 测试 UNet256 网络
    net256 = UNet256(8, device).to(device)
    # def forward(self, zt, ic, person_pose_embedding, garment_pose_embedding, time_step, noise_level):
    # 输入通道 9 是因为三组图片 concat： Itr128、Ia、Zt，三个图片 concat 为 Zt
    out256 = net256(torch.randn(4, 9, 256, 256).to(device),
                    torch.randn(4, 3, 256, 256).to(device),
                    torch.randn(4, 8).to(device),
                    torch.randn(4, 8).to(device),
                    time_step.to(device),
                    0.3)

    print("out256: ", out256.size())
