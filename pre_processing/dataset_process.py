"""
这个脚本是实现数据集不重复两两配对，并将配对后的一对图像重新命名饭别放置在两个文件夹中
可以产生大量的配对数据来使用，谨慎！！！

例如: 
1.jpg, 2.jpg, 3.jpg, 4.jpg
可以分别配对(1.jpg, 2.jpg), (1.jpg, 3.jpg), (1.jpg, 4.jpg), (2.jpg, 3.jpg), (2.jpg, 4.jpg), (3.jpg, 4.jpg)
把他们依次重新进行命名(000001.jpg, 000001.jpg), (000002.jpg, 000002.jpg), (000003.jpg, 000003.jpg), (000004.jpg, 000004.jpg), (000005.jpg, 000005.jpg), (000006.jpg, 000006.jpg) 并且分别把配对好的图片放置在文件夹a和b中
"""

import os
import random
import shutil
from itertools import combinations


def pair_and_copy_files(source_folder, dest_folder_a, dest_folder_b):
    # 两两配对

    os.makedirs(dest_folder_a, exist_ok=True)
    os.makedirs(dest_folder_b, exist_ok=True)

    # 遍历文件夹中的文件
    files = [f for f in os.listdir(source_folder) if f.endswith('.jpg')]

    # Generate all possible pairs
    pairs = list(combinations(files, 2))

    # 定义开始的文件名序列号
    for i, (file1, file2) in enumerate(pairs, start=1719):
        new_name = f"{i:07}.jpg"
        shutil.copy(os.path.join(source_folder, file1),
                    os.path.join(dest_folder_a, new_name))
        shutil.copy(os.path.join(source_folder, file2),
                    os.path.join(dest_folder_b, new_name))

    return pairs


def rename_files_in_folder(folder_path):
    # 打乱配对

    # 第一步：随机重命名文件
    files = os.listdir(folder_path)
    random.shuffle(files)
    temp_folder = os.path.join(folder_path, "temp")
    os.makedirs(temp_folder, exist_ok=True)

    used_names = set()
    for file in files:
        original_file_path = os.path.join(folder_path, file)
        while True:
            random_name = f"temp_{random.randint(1, 1000000000)}.jpg"
            if random_name not in used_names:
                used_names.add(random_name)
                break
        random_file_path = os.path.join(temp_folder, random_name)
        shutil.move(original_file_path, random_file_path)

    # 第二步：按顺序重命名文件
    temp_files = os.listdir(temp_folder)
    temp_files.sort()

    # 定义开始的文件名序列号
    for i, file in enumerate(temp_files, start=9567):
        new_file_name = f"{i:07}.jpg"
        original_file_path = os.path.join(temp_folder, file)
        new_file_path = os.path.join(folder_path, new_file_name)
        shutil.move(original_file_path, new_file_path)

    # 清理：删除临时文件夹
    shutil.rmtree(temp_folder)  # 使用 rmtree 删除非空目录


if __name__ == '__main__':
    # 执行两两配对
    source_folder = "/home/xkmb/tryondiffusion/data/val/ip2_raw"
    dest_folder_a = "/home/xkmb/tryondiffusion/data/val/ip2"  # The first new folder
    dest_folder_b = "/home/xkmb/tryondiffusion/data/val/ig2"  # The second new folder
    # pair_and_copy_files(source_folder, dest_folder_a, dest_folder_b)

    # 执行打乱配对，区别于两两配对
    folder_path = "/home/xkmb/tryondiffusion/data/train/ip"
    rename_files_in_folder(folder_path)
