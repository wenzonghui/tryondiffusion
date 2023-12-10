# -*- coding: utf-8 -*-
# @Author        : wenzonghui
# @Email         : wenzonghui@gmail.com
# @Create Date   : 2023-11-04 22:18:15
# @Last Modified : 2023-11-04 22:18:15
# @Description   :

import torch
from torch import nn
from pre_processing.person_pose_embedding.network import AutoEncoder as PersonAutoEncoder
from pre_processing.garment_pose_embedding.network import AutoEncoder as GarmentAutoEncoder
from network_elem import UNetBlockNoAttention, DownSample, UNetBlockAttention, UpSample, SinusoidalPosEmbed, AttentionPool1d


class UNet128(nn.Module):

    def __init__(self, pose_embed_len_dim, time_dim=256, noise_level=0.3):
        super().__init__()

        self.pos_encod_layer = SinusoidalPosEmbed(time_dim)

        # process clip embeddings
        self.attn_pool_layer = AttentionPool1d(pose_embed_len_dim,noise_level=noise_level)

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> person UNet >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # initial image embedding size 128x128 and 6 channels(concatenate rgb agnostic image and noise)

        # 3x3 conv layer person
        # person UNet：这里之所以是 6：因为 Ia 和 Zt 进行 concat 得到 Zt，通道变为 6
        self.init_conv_person = nn.Conv2d(6, 128, (3, 3), padding=1)

        # 128 unit encoder person
        self.block1_person = UNetBlockNoAttention(block_channel=128, clip_dim=pose_embed_len_dim, res_blocks_number=3)
        self.downsample1_person = DownSample(dim=128, dim_out=256, t_emb_dim=time_dim)

        # 64 unit encoder person
        self.block2_person = UNetBlockNoAttention(block_channel=256, clip_dim=pose_embed_len_dim, res_blocks_number=4)
        self.downsample2_person = DownSample(dim=256, dim_out=512, t_emb_dim=time_dim)

        # 32 unit encoder person
        self.block3_person = UNetBlockAttention(block_channel=512, clip_dim=pose_embed_len_dim, res_blocks_number=6, hw_dim=32)
        self.downsample3_person = DownSample(dim=512, dim_out=1024, t_emb_dim=time_dim)

        # 16 unit encoder person
        self.block4_person = UNetBlockAttention(block_channel=1024, clip_dim=pose_embed_len_dim, res_blocks_number=7, hw_dim=16)

        # 16 unit decoder person
        self.block5_person = UNetBlockAttention(block_channel=1024, clip_dim=pose_embed_len_dim, res_blocks_number=7, hw_dim=16, input_channel=1024 * 2)
        self.upsample1_person = UpSample(dim=1024, dim_out=512, t_emb_dim=time_dim)

        # 32 unit decoder person
        self.block6_person = UNetBlockAttention(block_channel=512, clip_dim=pose_embed_len_dim, res_blocks_number=6, hw_dim=32, input_channel=512 * 2)
        self.upsample2_person = UpSample(dim=512, dim_out=256, t_emb_dim=time_dim)

        # 64 unit decoder person
        self.block7_person = UNetBlockNoAttention(block_channel=256, clip_dim=pose_embed_len_dim, res_blocks_number=4, input_channel=256 * 2)
        self.upsample3_person = UpSample(dim=256, dim_out=128, t_emb_dim=time_dim)

        # 128 unit decoder person
        self.block8_person = UNetBlockNoAttention(block_channel=128, clip_dim=pose_embed_len_dim, res_blocks_number=3, input_channel=128 * 2)

        self.final_conv_person = nn.Conv2d(128, 3, (3, 3), padding=1)

        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< person UNet ends <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> garment UNet >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # initial image embedding size 128x128 and 3 channels

        # 3x3 conv layer garment
        # garment UNet 输入 ic, 所以通道是3
        self.init_conv_garment = nn.Conv2d(3, 128, (3, 3), padding=1)

        # 128 unit encoder garment
        self.block1_garment = UNetBlockNoAttention(block_channel=128, clip_dim=pose_embed_len_dim, res_blocks_number=3)
        self.downsample1_garment = DownSample(dim=128, dim_out=256, t_emb_dim=time_dim)

        # 64 unit encoder garment
        self.block2_garment = UNetBlockNoAttention(block_channel=256, clip_dim=pose_embed_len_dim, res_blocks_number=4)
        self.downsample2_garment = DownSample(dim=256, dim_out=512, t_emb_dim=time_dim)

        # 32 unit encoder garment
        self.block3_garment = UNetBlockNoAttention(block_channel=512, clip_dim=pose_embed_len_dim, res_blocks_number=6)
        self.downsample3_garment = DownSample(dim=512, dim_out=1024, t_emb_dim=time_dim)

        # 16 unit encoder garment
        self.block4_garment = UNetBlockNoAttention(block_channel=1024, clip_dim=pose_embed_len_dim, res_blocks_number=7)

        # 16 unit decoder garment
        self.block5_garment = UNetBlockNoAttention(block_channel=1024, clip_dim=pose_embed_len_dim, res_blocks_number=7, input_channel=1024 * 2)
        self.upsample1_garment = UpSample(dim=1024, dim_out=512, t_emb_dim=time_dim)

        # 32 unit decoder garment
        self.block6_garment = UNetBlockNoAttention(block_channel=512, clip_dim=pose_embed_len_dim, res_blocks_number=6, input_channel=512 * 2)

        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< garment UNet ends <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    def forward(self, zt, ic, person_pose_embedding, garment_pose_embedding, time_step):
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
        pose_embeddings = torch.concat((person_pose_embedding[:, None, :], garment_pose_embedding[:, None, :]), dim=1)

        # get clip embeddings
        clip_embedding = self.attn_pool_layer(pose_embeddings, time_step)

        # passing through initial conv layer: ic
        # out: 128 * 128 * 128
        ic_init = self.init_conv_garment(ic)

        # passing through initial conv layer: zt
        # out: 128 * 128 * 128
        zt_init = self.init_conv_person(zt)

        # entering UNet
        # 128 - garment
        # out: 128 * 128 * 128
        ic_128 = self.block1_garment(ic_init, clip_embedding)

        # 128 - person
        # out: 128 * 128 * 128
        zt_128_1 = self.block1_person(zt_init, clip_embedding)

        # 64 - garment
        # out: 64 * 64 * 256
        ic_64_downsample = self.downsample1_garment(ic_128, pos_encoding)
        ic_64 = self.block2_garment(ic_64_downsample, clip_embedding)

        # 64 - person:
        # out: 64 * 64 * 256
        zt_64_1_downsample = self.downsample1_person(zt_128_1, pos_encoding)
        zt_64_1 = self.block2_person(zt_64_1_downsample, clip_embedding)

        # 32 - garment
        # out: 32 * 32 * 512
        ic_32_1_downsample = self.downsample2_garment(ic_64, pos_encoding)
        ic_32_1 = self.block3_garment(ic_32_1_downsample, clip_embedding)

        # 32 - person
        # out: 32 * 32 * 512
        zt_32_1_downsample = self.downsample2_person(zt_64_1, pos_encoding)
        zt_32_1 = self.block3_person(zt_32_1_downsample, clip_embedding, pose_embeddings, ic_32_1)

        # 16 - garment
        # out: 16 * 16 * 1024
        ic_16_1_downsample = self.downsample3_garment(ic_32_1, pos_encoding)
        ic_16_1 = self.block4_garment(ic_16_1_downsample, clip_embedding)

        # 16 - person
        # out: 16 * 16 * 1024
        zt_16_1_downsample = self.downsample3_person(zt_32_1, pos_encoding)
        zt_16_1 = self.block4_person(zt_16_1_downsample, clip_embedding, pose_embeddings, ic_16_1)

        # --------------------------------------------- 右边一半 ---------------------------------------------
        # 16 - garment
        # out: 16 * 16 * 1024
        ic_16_2 = self.block5_garment(ic_16_1, clip_embedding, ic_16_1)

        # 16 - person
        # out: 16 * 16 * 1024
        zt_16_2 = self.block5_person(zt_16_1, clip_embedding, pose_embeddings, ic_16_2, zt_16_1)

        # 32 - garment
        # out: 32 * 32 * 512
        ic_32_2_upsample = self.upsample1_garment(ic_16_2, pos_encoding)
        ic_32_2 = self.block6_garment(ic_32_2_upsample, clip_embedding, ic_32_1)

        # 32 - person
        # out: 32 * 32 * 512
        zt_32_2_upsample = self.upsample1_person(zt_16_2, pos_encoding)
        zt_32_2 = self.block6_person(zt_32_2_upsample, clip_embedding, pose_embeddings, ic_32_2, zt_32_1)

        # 64 - person
        # out: 64 * 64 * 256
        zt_64_2_upsample = self.upsample2_person(zt_32_2, pos_encoding)
        zt_64_2 = self.block7_person(zt_64_2_upsample, clip_embedding, zt_64_1)

        # 128 - person
        # out: 128 * 128 * 128
        zt_128_2_upsample = self.upsample3_person(zt_64_2, pos_encoding)
        zt_128_2 = self.block8_person(zt_128_2_upsample, clip_embedding, zt_128_1)

        # final conv layer - person
        zt_final = self.final_conv_person(zt_128_2)

        return zt_final
    

if __name__ == "__main__":
    device = torch.device("cpu")
    # 测试 UNet128 网络
    net128 = UNet128(pose_embed_len_dim=8).to(device)
    fc1 = PersonAutoEncoder(34)
    fc2 = GarmentAutoEncoder(34)
    # def forward(self, zt, ic, person_pose_embedding, garment_pose_embedding, time_step, noise_level):
    # 这里之所以是 6：因为 Ia 和 Zt 进行 concat，通道变为 6

    # 输入模型的样本
    zt_sample = torch.randn(4, 6, 128, 128).to(device)
    ic_sample = torch.randn(4, 3, 128, 128).to(device)
    pose_embedding_sample = torch.randn(4, 8).to(device)
    time_step_sample = torch.randint(low=1, high=1000, size=(4,)).to(device)
    x = torch.rand(1, 34)
    # out128 = net128(zt_sample, ic_sample, pose_embedding_sample, pose_embedding_sample, time_step_sample)
    # print("out128: ", out128.size())
    
    # 计算网络参数量
    from thop import profile
    macs_unet128, params_unet128 = profile(net128, inputs=(zt_sample,ic_sample,pose_embedding_sample, pose_embedding_sample, time_step_sample))
    print('UNet128 Params: ', params_unet128)

    macs_fc1, params_fc1 = profile(fc1, inputs=(x))
    print('FC1 Params: ', params_fc1)

    macs_fc2, params_fc2 = profile(fc2, inputs=(x))
    print('FC2 Params: ', params_fc2)

    print("Total Params: ", params_unet128 + params_fc1 + params_fc2)
