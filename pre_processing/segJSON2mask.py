'''
对于手动通过 segment anything 进行分割得到的 json 文件，可以通过此脚本得到固定颜色区块的 mask 图像
进一步可以通过 mask 图像在原图中把特定部位分割出来
'''

import json
import os
import numpy as np
import cv2


def generate_random_color():
    # 设置随机颜色，这里暂时用不到
    return tuple(np.random.choice(range(256), size=3).tolist())


def generate_mask(json_file_path, image_file_path, mask_image_save_path):
    # 通过分割信息 json 生成单张 mask 图像

    # Load the JSON file
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # Load the original image to get its dimensions
    original_image = cv2.imread(image_file_path)
    height, width, _ = original_image.shape
    image_name_with_extension = os.path.basename(
        image_file_path)  # 获取文件名，包括扩展名
    image_name = os.path.splitext(image_name_with_extension)[0]  # 去除扩展名

    # Create an empty mask with 3 channels to hold colored masks (BGR format)
    color_mask = np.zeros((height, width, 3), dtype=np.uint8)

    # Extract the 'objects' data from the JSON
    objects_structure = data.get('objects', [])

    # Iterate over each object and draw its segmentation with a unique color
    for obj in objects_structure:

        category = obj.get('category')
        segmentation = obj.get('segmentation', [])
        if segmentation and category == "up":
            # 处理上装，固定颜色为红色，二进制为255
            # Convert segmentation to a numpy array and make sure it's an integer array
            points = np.array(segmentation, dtype=np.int32)
            # Generate a random color for each object：BGR
            color = (0, 0, 255)
            # Draw the polygon on the mask
            cv2.fillPoly(color_mask, [points], color)
        elif segmentation and category == "down":
            # 处理下装，固定颜色为青色，二进制为510
            # Convert segmentation to a numpy array and make sure it's an integer array
            points = np.array(segmentation, dtype=np.int32)
            # Generate a random color for each object
            color = (255, 255, 0)
            # Draw the polygon on the mask
            cv2.fillPoly(color_mask, [points], color)
        elif segmentation and category == "person":
            # 处理人体，固定颜色为米色，二进制为710
            # Convert segmentation to a numpy array and make sure it's an integer array
            points = np.array(segmentation, dtype=np.int32)
            # Generate a random color for each object
            color = (220, 245, 245)
            # Draw the polygon on the mask
            cv2.fillPoly(color_mask, [points], color)
        elif segmentation and category == "__background__":
            # 处理背景，固定颜色为绿色，二进制为376
            # Convert segmentation to a numpy array and make sure it's an integer array
            points = np.array(segmentation, dtype=np.int32)
            # Generate a random color for each object
            color = (0, 252, 124)
            # Draw the polygon on the mask
            cv2.fillPoly(color_mask, [points], color)

    # 保存图像
    mask_image_path = os.path.join(mask_image_save_path, f'{image_name}.jpg')
    cv2.imwrite(mask_image_path, color_mask)
    return mask_image_path


def generate_masks(json_file_dir, image_file_dir, mask_image_save_dir):
    # 批量处理，从分割信息 json 到 mask 图像

    # 获取 json 文件夹中的文件列表并按文件名排序
    json_files = sorted(os.listdir(json_file_dir))

    # 获取 image 文件夹中的文件列表并按文件名排序
    image_files = sorted(os.listdir(image_file_dir))

    # 确保目标目录存在
    os.makedirs(mask_image_save_dir, exist_ok=True)

    # 遍历文件列表并处理每个文件对
    for json_file, image_file in zip(json_files, image_files):
        print(json_file, image_file)

        # 构建完整的文件路径
        json_file_path = os.path.join(json_file_dir, json_file)
        image_file_path = os.path.join(image_file_dir, image_file)

        # 调用生成 mask 的函数
        generate_mask(json_file_path, image_file_path, mask_image_save_dir)

    print("处理完成！")


if __name__ == '__main__':
    json_file_dir = '/home/xkmb/tryondiffusion/data_test/ip_json'
    image_file_dir = '/home/xkmb/tryondiffusion/data_test/ip'
    mask_image_save_dir = '/home/xkmb/tryondiffusion/data_test/ip_mask'

    generate_masks(json_file_dir, image_file_dir, mask_image_save_dir)
