import json
import os
import time
import cv2
import torch
from mmpose_main.inference import pose_model_process
from segment_anything import sam_model_registry
from pre_processing.generate_mask import seg_anyone, seg_bg, seg_person_left_arm, seg_person_right_arm, seg_up
from pre_processing.person_pose_embedding.utils.dataloader import normalize_lst
from pre_processing.pose_json_preprocess import start_pose_json_process
from pre_processing.segmented_garment import start_seg_garment
from pre_processing.segmented_person import start_seg_person
from utils.dataloader_train import create_transforms_imgs
from utils.utils import read_img
from diffusion import Diffusion, smoothen_image


def pre_process(person, garment):
    # 推理前数据预处理，得到的是对应数据的路径
    # 现在设计方法：person.jpg, garment.jpg, person.json, garment.json在同一个文件夹下，只需要给到 person.jpg 和 garment.jpg 的路径即可
    # 这里的 person.json, garment.json 是使用分割工具得到的 mask 图的 json

    task_time = time.time()
    person_name = person.split('/')[-1].split('.')[0]
    garment_name = garment.split('/')[-1].split('.')[0]
    task_name = f"{task_time}_{person_name}_{garment_name}"
    output_dir = f'inference/{task_name}'
    mask_output_dir = f'inference/{task_name}/mask'
    print(f"Save in {output_dir}")

    # 人物通过 mmpose 姿态识别后得到的 json 文件路径
    jp_json = pose_model_process(person, output_dir)
    jp_json_normlize = start_pose_json_process(jp_json, output_dir)  # mmpose 得到的 json 文件后再通过脚本进行规则化得到的 json

    # 衣服通过 mmpose 姿态识别后得到的 json 文件路径
    jg_json = pose_model_process(garment, output_dir)
    jg_json_normlize = start_pose_json_process(jg_json, output_dir)  # mmpose 得到的 json 文件后再通过脚本进行规则化得到的 json

    # 加载分割模型
    sam_checkpoint = "/home/xkmb/tryondiffusion/models/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device="cuda")

    # 分割得到去掉衣服的人的图像
    # 这里 an_mask_image_path 和 an_image_path 都是中间变量，临时存放图像
    an_mask_image_path = seg_anyone(person, jp_json, mask_output_dir, seg_up, sam)  # 生成一张衣服的 mask 图
    an_image_path = start_seg_person(person, an_mask_image_path, output_dir)  # 去掉衣服的图像
    an_mask_image_path = seg_anyone(an_image_path, jp_json, mask_output_dir, seg_bg, sam)  # 生成一张背景的 mask 图
    an_image_path = start_seg_person(an_image_path, an_mask_image_path, output_dir)  # 去掉背景的图像
    an_mask_image_path = seg_anyone(an_image_path, jp_json, mask_output_dir, seg_person_left_arm, sam)  # 生成一张左胳膊的 mask 图
    an_image_path = start_seg_person(an_image_path, an_mask_image_path, output_dir)  # 去掉左胳膊的图像
    an_mask_image_path = seg_anyone(an_image_path, jp_json, mask_output_dir, seg_person_right_arm, sam)  # 生成一张右胳膊的 mask 图
    ia = start_seg_person(an_image_path, an_mask_image_path, output_dir)  # 去掉右胳膊的图像

    # 分割得到衣服的图像
    ic_mask_path = seg_anyone(garment, jg_json, mask_output_dir, seg_up, sam)  # 生成一张衣服的 mask 图
    ic = start_seg_garment(garment, ic_mask_path, output_dir)  # 扣出的衣服的图像

    # 这里返回的都是对应文件的路径
    return person, garment, ia, ic, jp_json_normlize, jg_json_normlize, task_name


def inference_unet128(args, person_image_path, garment_image_path, model):
    # 加载模型
    # model.load_state_dict(torch.load('tmp_models/ckpt128/ckpt_0.pt'))
    # model.eval()

    # 准备输入数据
    # 假设 pre_process 函数能够正确处理输入图片，并返回模型需要的所有输入参数
    ip, ig, ia, ic, jp, jg, task_name = pre_process(person_image_path, garment_image_path)

    ia = smoothen_image(create_transforms_imgs(read_img(ia), args.unet_dim).unsqueeze(0).to(args.device), args.sigma, args.device)
    ic = smoothen_image(create_transforms_imgs(read_img(ic), args.unet_dim).unsqueeze(0).to(args.device), args.sigma, args.device)
    ip = create_transforms_imgs(read_img(ip), args.unet_dim).unsqueeze(0).to(args.device)
    ig = create_transforms_imgs(read_img(ig), args.unet_dim).unsqueeze(0).to(args.device)

    jp_data = []
    with open(jp, 'r') as jp:
        jp_json = json.load(jp)
        jp_json_normalize = normalize_lst(jp_json)
        jp_data.append(jp_json_normalize)
    jp_tensor = torch.tensor(jp_data)
    jp_fc1 = model.fc1(jp_tensor)
    jp = jp_fc1[1]
    
    jg_data = []
    with open(jg, 'r') as jg:
        jg_json = json.load(jg)
        jg_json_normalize = normalize_lst(jg_json)
        jg_data.append(jg_json_normalize)
    jg_tensor = torch.tensor(jg_data)
    jg_fc2 = model.fc2(jg_tensor)
    jg = jg_fc2[1]

    # sampled image
    sampled_image = model.sample(use_ema=False, conditional_inputs=(ia, ic, jp, jg))
    sampled_image = sampled_image[0].permute(1, 2, 0).squeeze().cpu().numpy()

    # ema sampled image
    # ema_sampled_image = model.sample(use_ema=True, conditional_inputs=(ia, ic, jp, jg))
    # ema_sampled_image = ema_sampled_image[0].permute(1, 2, 0).squeeze().cpu().numpy()

    # base images
    ip_np = ip.squeeze(0).permute(1, 2, 0).squeeze().cpu().numpy()
    ig_np = ig.squeeze(0).permute(1, 2, 0).squeeze().cpu().numpy()
    ic_np = ic.squeeze(0).permute(1, 2, 0).squeeze().cpu().numpy()
    ia_np = ia.squeeze(0).permute(1, 2, 0).squeeze().cpu().numpy()

    # make to folders
    os.makedirs(os.path.join("inference", f"{task_name}"), exist_ok=True)
    # define folder paths
    save_folder = os.path.join("inference", f"{task_name}")

    # 保存生成的图像
    cv2.imwrite(os.path.join(save_folder, "person.jpg"), ip_np)
    cv2.imwrite(os.path.join(save_folder, "garment.jpg"), ig_np)
    cv2.imwrite(os.path.join(save_folder, "seg_garment.jpg"), ic_np)
    cv2.imwrite(os.path.join(save_folder, "seg_person.jpg"), ia_np)
    # save sampled image
    cv2.imwrite(os.path.join(save_folder, "itr128.jpg"), sampled_image)
    # save ema sampled image
    # cv2.imwrite(os.path.join(save_folder, "itr128_ema.jpg"), ema_sampled_image)
    print(f"In inference: Saved images")

    itr128_path = os.path.join(save_folder, f"itr128.jpg")
    return itr128_path


def inference_unet256(person, garment, unet256_diffusion_model, output_dir):
    # unet256 推理
    pass


def inference_sr_diffusion(itr256_path, output_dir):
    # sr1024 推理
    pass



class InferenceArgParser:
    def __init__(self):
        self.device = 'cuda'
        self.use_mix_precision = False
        self.unet_dim = 128
        self.sigma = float(torch.FloatTensor(1).uniform_(0.4, 0.6))
        self.model_path = '/home/xkmb/tryondiffusion/models/ckpt_40.pth'  # 模型文件的路径
        self.fc1_model_path = '/home/xkmb/tryondiffusion/models/fc1.pth'  # FC1模型的路径
        self.fc2_model_path = '/home/xkmb/tryondiffusion/models/fc2.pth'  # FC2模型的路径


if __name__ == "__main__":
    inference_args = InferenceArgParser()

    # 创建 Diffusion 模型
    unet128_diffusion_model = Diffusion(device="cuda", unet_dim=inference_args.unet_dim, pose_embed_dim=8, time_steps=256, beta_start=1e-4, beta_end=0.02, beta_ema=0.995, noise_input_channel=3)

    unet128_diffusion_model.prepare_for_inference(inference_args)

    person_path = 'data/test/person.jpg'
    garment_path = 'data/test/garment.jpg'
    
    # 调用 inference 函数进行推理并保存结果图像
    inference_unet128(inference_args, person_path, garment_path, unet128_diffusion_model)
