"""
通过 sam 得到的分割 mask 图像，与原图像一起进行处理，进而得到分割出的 person 的图像
"""

import cv2
import numpy as np
from segment_anything import sam_model_registry, SamPredictor

from generate_mask import seg_any, seg_anyone, seg_up

def get_upper_garment(img, img_parse_map):
    sum_img_parse_map = np.sum(img_parse_map, axis=2)
    # print(sum_img_parse_map[500][300:])

    for i in range(sum_img_parse_map.shape[0]):
        for j in range(sum_img_parse_map.shape[1]):
            if 200 <= sum_img_parse_map[i][j] <= 300 or sum_img_parse_map[i][j] > 400:  # 这个二进制的值需要打印出来看在哪个值的附近
                sum_img_parse_map[i][j] = 1
            else:
                sum_img_parse_map[i][j] = 0

    upper_garment_segment = (sum_img_parse_map.reshape(*sum_img_parse_map.shape, 1) * img).astype(np.uint16)

    return upper_garment_segment

def start_seg_garment(an_image_path, an_mask_image_path, output_dir):
    # 处理单张图片，用于推理

    import os
    import sys

    # 如果输入的不是图片，那么不进行处理
    if not an_image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
        return
    if not an_mask_image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
        return

    parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, parentdir)

    from utils.utils import read_img

    # 如果不存在输出路径，则递归的创建它的文件夹
    os.makedirs(output_dir, exist_ok=True)

    # 打开并处理图片
    image = read_img(an_image_path)
    image_mask = read_img(an_mask_image_path)
    segmented_garment = get_upper_garment(image, image_mask)

    # 保存处理过的图片到目标文件夹
    filename = an_image_path.split('/')[-1].split('.')[0]
    segmented_garment_path = os.path.join(output_dir, f"seg_garment_raw.jpg")
    # Save the segmented image with the specified filename
    cv2.imwrite(segmented_garment_path, segmented_garment)

    print('图片处理完成。')
    # 返回分割出的 person 的图路径
    return segmented_garment_path

def start_seg_garments(input_dir1, input_dir2, output_dir):
    # 批量化的处理，处理文件夹

    import os
    import sys

    parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, parentdir)

    from utils.utils import read_img

    ig_dir = input_dir1
    mask_dir = input_dir2
    ic_dir = output_dir
    # 如果不存在输出路径，则递归的创建它的文件夹
    os.makedirs(ic_dir, exist_ok=True)

    # 遍历源文件夹中的所有文件
    for filename in os.listdir(ig_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):  # 检查文件是否为图片
            img_path = os.path.join(ig_dir, filename)
            mask_path = os.path.join(mask_dir, filename)
            ic_path = os.path.join(ic_dir, filename)

            # 打开并处理图片
            image = read_img(img_path)
            image_mask = read_img(mask_path)
            segmented_person = get_upper_garment(image, image_mask)

            # 保存处理过的图片到目标文件夹
            cv2.imwrite(ic_path, segmented_person)

    print('图片处理完成。')


if __name__ == "__main__":
    # 加载模型
    sam_checkpoint = "/home/xkmb/tryondiffusion/models/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device="cuda")


    # 用于批量操作
    # 几张图片测试一下
    # image_dir = '/home/xkmb/tryondiffusion/segment_anything_main/test/images'
    # json_dir = '/home/xkmb/tryondiffusion/segment_anything_main/test/pose_json'
    # mask_dir = '/home/xkmb/tryondiffusion/segment_anything_main/test/mask'
    # output_dir = "/home/xkmb/tryondiffusion/segment_anything_main/test/ic"
    # 正式执行分割任务
    image_dir = '/home/xkmb/data/new_val_ip/ia'
    json_dir = '/home/xkmb/pose_output/val/ip/predictions'
    mask_dir = '/home/xkmb/data/new_val_ip/mask'
    output_dir = "/home/xkmb/data/new_val_ip/new_ia"
    
    # seg_any(image_dir, json_dir, mask_dir, seg_up, sam)
    start_seg_garments(image_dir, mask_dir, output_dir)


    # 用于单张图片推理
    # 推理时候 mask 放在输出文件夹下的子目录，防止文件名重复覆盖掉原文件
    # an_image_path = '/home/xkmb/tryondiffusion/segment_anything_main/test/images/0000001.jpg'
    # a_json_path = '/home/xkmb/tryondiffusion/segment_anything_main/test/pose_json/0000001.json'
    # mask_dir = '/home/xkmb/tryondiffusion/segment_anything_main/test/mask'
    # output_dir = "/home/xkmb/tryondiffusion/segment_anything_main/test/ic"
    # an_mask_image_path = seg_anyone(an_image_path, a_json_path, mask_dir, seg_up, sam)
    # start_seg_garment(an_image_path, an_mask_image_path, output_dir)
