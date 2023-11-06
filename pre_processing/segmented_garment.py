import cv2
import numpy as np


def get_upper_garment(img, img_parse_map):
    sum_img_parse_map = np.sum(img_parse_map, axis=2)
    sum_img_parse_map[sum_img_parse_map != 339] = 0
    sum_img_parse_map[sum_img_parse_map == 339] = 1
    upper_garment_segment = (sum_img_parse_map.reshape(*sum_img_parse_map.shape, 1) * img).astype(np.uint8)
    return upper_garment_segment


if __name__ == "__main__":
    import os
    import sys

    parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, parentdir)

    from utils.utils import read_img

    # ig_dir = "/home/xkmb/tryondiffusion/data/train/ig"
    # ic_dir = "/home/xkmb/tryondiffusion/data/train/ic"
    # image_parse_dir = "/home/xkmb/tryondiffusion/data/train/image-parse-v3"

    ig_dir = "./data/ig"
    ic_dir = "./data/ic"
    image_parse_dir = "./data/image-parse-v3"

    # 遍历源文件夹中的所有文件
    for filename in os.listdir(ig_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):  # 检查文件是否为图片
            img_path = os.path.join(ig_dir, filename)
            img_parse_path = os.path.join(image_parse_dir, filename[:-3] + 'png')
            ic_path = os.path.join(ic_dir, filename)
            # print(img_path)
            # print(img_parse_path)
            # print(ic_path)

            # 打开并处理图片
            image = read_img(img_path)
            image_parse = read_img(img_parse_path)

            segmented_garment = get_upper_garment(image, image_parse)

            # 保存处理过的图片到目标文件夹
            cv2.imwrite(ic_path, segmented_garment)

    print('图片处理完成。')
