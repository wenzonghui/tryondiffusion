"""
先利用程序得到关键点坐标点
再利用 sam 坐标点输入进行资产分割
"""

import json
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamPredictor

def calculate_intersection(A, B, C, D):
    # 计算线段 AC 的斜率和截距
    m_AC = (C[1] - A[1]) / (C[0] - A[0]) if (C[0] - A[0]) != 0 else float('inf')
    c_AC = A[1] - m_AC * A[0] if m_AC != float('inf') else A[0]  # 对垂直线进行处理

    # 计算线段 BD 的斜率和截距
    m_BD = (D[1] - B[1]) / (D[0] - B[0]) if (D[0] - B[0]) != 0 else float('inf')
    c_BD = B[1] - m_BD * B[0] if m_BD != float('inf') else B[0]  # 对垂直线进行处理

    # 解方程找到交点
    if m_AC != m_BD:  # 确保不是平行线
        if m_AC != float('inf') and m_BD != float('inf'):
            x = (c_BD - c_AC) / (m_AC - m_BD)
            y = m_AC * x + c_AC
        elif m_AC == float('inf'):
            x = c_AC
            y = m_BD * x + c_BD
        else:  # m_BD == float('inf')
            x = c_BD
            y = m_AC * x + c_AC
        return [x, y]
    else:
        return None  # 平行线没有交点

def get_points(json_path):
    # 利用mmpose得到的关键点坐标，通过程序处理得到7个关键点：鼻子1，左眼睛2，右眼睛3，左肩膀6，右肩膀7，左大腿根12， 右大腿根13
    # 进一步处理得到6个关键点：鼻子1，左眼睛2，右眼睛3，左胸，右胸，肚子

    # 打开并读取 JSON 文件
    with open(json_path, 'r') as file:
        data = json.load(file)

    # 提取 keypoints 的鼻子1，左眼睛2，右眼睛3，左肩膀6，右肩膀7，左大腿根12， 右大腿根13
    nose_point = data[0]["keypoints"][0]
    left_eye_point = data[0]["keypoints"][1]
    right_eye_point = data[0]["keypoints"][2]
    left_shoulder_point = data[0]["keypoints"][5]
    right_shoulder_point = data[0]["keypoints"][6]
    left_legend_point = data[0]["keypoints"][11]
    right_legend_point = data[0]["keypoints"][12]
    left_elbow_point = data[0]["keypoints"][7]
    left_wrist_point = data[0]["keypoints"][9]
    right_elbow_point = data[0]["keypoints"][8]
    right_wrist_point = data[0]["keypoints"][10]

    # 保留鼻子1，左眼睛2，右眼睛3，左胸，右胸，交叉点（肚子），左肩6，左肘8，左腕10，右肩7，右肘9，右腕11
    intersection_point = calculate_intersection(right_shoulder_point, left_shoulder_point, left_legend_point, right_legend_point)
    left_chest_point = [left_shoulder_point[0]+(intersection_point[0]-left_shoulder_point[0])/2, left_shoulder_point[1]+(intersection_point[1]-left_shoulder_point[1])/2]
    right_chest_point = [intersection_point[0]+(right_shoulder_point[0]-intersection_point[0])/2, right_shoulder_point[1]+(intersection_point[1]-right_shoulder_point[1])/2]
    
    final_points = []
    final_points.append(nose_point)  # 0
    final_points.append(left_eye_point)  # 1
    final_points.append(right_eye_point)  # 2
    final_points.append(left_chest_point)  # 3
    final_points.append(right_chest_point)  # 4
    final_points.append(intersection_point)  # 5
    final_points.append(left_shoulder_point)  # 6
    final_points.append(left_elbow_point)  # 7
    final_points.append(left_wrist_point)  # 8
    final_points.append(right_shoulder_point)  # 9
    final_points.append(right_elbow_point)  # 10
    final_points.append(right_wrist_point)  # 11
    # print(final_points)
    
    return final_points

def seg_up(image_path, json_path, output_dir, sam):
    # 对于上衣的分割，采用三个正向点定位 + 三个反向点排除的方法
    # 三个正向点：左胸，右胸，肚子
    # 三个反向点：鼻子1，左眼睛2，右眼睛3

    from segment_anything import sam_model_registry, SamPredictor

    # 获取关键点
    ups_points = get_points(json_path)
    nose_point = ups_points[0]
    left_eye_point = ups_points[1]
    right_eye_point = ups_points[2]
    left_chest_point = ups_points[3]
    right_chest_point = ups_points[4]
    belly_point = ups_points[5]
    left_shoulder_point = ups_points[6]
    right_shoulder_point = ups_points[9]

    # 打开图片
    print(image_path)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 加载模型处理关键点
    # sam_checkpoint = "/home/xkmb/tryondiffusion/segment_anything_main/checkpoints/sam_vit_h_4b8939.pth"
    # sam_hq_checkpoint = "/home/xkmb/tryondiffusion/segment_anything_main/checkpoints/sam_hq_vit_h.pth"
    # model_type = "vit_h"
    # sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    # sam.to(device="cuda")
    predictor = SamPredictor(sam)
    predictor.set_image(image)

    input_point = np.array([nose_point, left_eye_point, right_eye_point, left_chest_point, right_chest_point, belly_point])
    input_label = np.array([0, 0, 0, 1, 1, 1])

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )

    mask_input = logits[np.argmax(scores), :, :]

    masks, _, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        mask_input=mask_input[None, :, :],
        multimask_output=False,
    )

    # 确保 masks 是二维的
    if len(masks.shape) == 3 and masks.shape[0] == 1:  # 如果 masks 形状为 [1, H, W]
        masks = masks[0]  # 取出第一个通道

    # 绘制掩码
    # 创建一个与原图同尺寸的空白彩色画布
    colored_mask = np.zeros_like(image)
    # 填充选中和未选中的掩码区域
    colored_mask[masks > 0] = (255, 0, 0)  # 红色
    colored_mask[masks == 0] = (124, 252, 0)  # 绿色

    # 在原图上绘制掩码
    canvas = np.zeros_like(image)  # 创建一个空画布，用于绘制
    canvas = image.copy()  # 在画布上绘制原始图像
    canvas = cv2.addWeighted(canvas, 0.7, colored_mask, 0.3, 0)  # 调整原图与掩码的透明度比例并叠加

    # 是否绘制关键点
    # for point, label in zip(input_point, input_label):
    #     color = (0, 0, 0) if label == 1 else (0, 0, 0)  # 正向点绿色，反向点红色
    #     canvas = cv2.circle(canvas, (int(point[0]), int(point[1])), radius=5, color=color, thickness=-1)
    #     colored_mask = cv2.circle(colored_mask, (int(point[0]), int(point[1])), radius=5, color=color, thickness=-1)

    image_name = os.path.basename(image_path)[:-4]
    output_image_path = os.path.join(output_dir, f"{image_name}.jpg")

    # 只保存掩码图像
    colored_mask = cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR)  # 将图像转回 OpenCV 格式
    cv2.imwrite(output_image_path, colored_mask)

    # 保存原图与掩码的叠加图
    # canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)  # 将图像转回 OpenCV 格式
    # cv2.imwrite(output_image_path, canvas)

    # 返回分割后的图像路径
    return output_image_path
    
def seg_bg(image_path, json_path, output_dir, sam):
    # 对于衣服的分割，采用点选 + 框选的方法
    # 点选：背景（左上，右上，左下，右下）

    from segment_anything import sam_model_registry, SamPredictor

    # 获取关键点
    left_top_point = [10, 10]
    right_top_point = [758, 10]
    left_bottom_point = [10, 1014]
    right_bottom_point = [758, 1014]

    # 打开图片
    print(image_path)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 加载模型处理关键点
    # sam_checkpoint = "/home/xkmb/tryondiffusion/segment_anything_main/checkpoints/sam_vit_h_4b8939.pth"
    # sam_hq_checkpoint = "/home/xkmb/tryondiffusion/segment_anything_main/checkpoints/sam_hq_vit_h.pth"
    # model_type = "vit_h"
    # sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    # sam.to(device="cuda")
    predictor = SamPredictor(sam)
    predictor.set_image(image)

    # 分割用到的关键点
    input_point = np.array([left_top_point, right_top_point, left_bottom_point, right_bottom_point])
    input_label = np.array([1, 1, 1, 1])

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )

    mask_input = logits[np.argmax(scores), :, :]

    masks, _, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        mask_input=mask_input[None, :, :],
        multimask_output=False,
    )

    # 确保 masks 是二维的
    if len(masks.shape) == 3 and masks.shape[0] == 1:  # 如果 masks 形状为 [1, H, W]
        masks = masks[0]  # 取出第一个通道

    # 绘制掩码
    # 创建一个与原图同尺寸的空白彩色画布
    colored_mask = np.zeros_like(image)
    # 填充选中和未选中的掩码区域，只区分两种颜色
    colored_mask[masks > 0] = (255, 0, 0)  # （正向点）红色
    colored_mask[masks == 0] = (124, 252, 0)  # （负向点）绿色

    # 在原图上绘制掩码
    canvas = np.zeros_like(image)  # 创建一个空画布，用于绘制
    canvas = image.copy()  # 在画布上绘制原始图像
    canvas = cv2.addWeighted(canvas, 0.7, colored_mask, 0.3, 0)  # 调整原图与掩码的透明度比例并叠加

    # 是否绘制关键点
    # for point, label in zip(input_point, input_label):
    #     color = (0, 0, 0) if label == 1 else (0, 0, 0)  # 正向点绿色，反向点红色
    #     canvas = cv2.circle(canvas, (int(point[0]), int(point[1])), radius=5, color=color, thickness=-1)
    #     colored_mask = cv2.circle(colored_mask, (int(point[0]), int(point[1])), radius=5, color=color, thickness=-1)

    image_name = os.path.basename(image_path)[:-4]
    output_image_path = os.path.join(output_dir, f"{image_name}.jpg")

    # 只保存掩码图像
    colored_mask = cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR)  # 将图像转回 OpenCV 格式
    cv2.imwrite(output_image_path, colored_mask)

    # 保存原图与掩码的叠加图
    # canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)  # 将图像转回 OpenCV 格式
    # cv2.imwrite(output_image_path, canvas)

    # 返回分割后的图像路径
    return output_image_path

def seg_person_left_arm(image_path, json_path, output_dir, sam):
    # 对于衣服的分割，采用点选 + 框选的方法
    # 点选：左胳膊（左肩，左肘，左腕）
    # 框选：根据胳膊关键点去放大坐标框选胳膊

    from segment_anything import sam_model_registry, SamPredictor

    # 获取关键点
    ups_points = get_points(json_path)
    left_shoulder_point = ups_points[6]
    left_elbow_point = ups_points[7]
    left_wrist_point = ups_points[8]

    # 打开图片
    print(image_path)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 加载模型处理关键点
    # sam_checkpoint = "/home/xkmb/tryondiffusion/segment_anything_main/checkpoints/sam_vit_h_4b8939.pth"
    # sam_hq_checkpoint = "/home/xkmb/tryondiffusion/segment_anything_main/checkpoints/sam_hq_vit_h.pth"
    # model_type = "vit_h"
    # sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    # sam.to(device="cuda")
    predictor = SamPredictor(sam)
    predictor.set_image(image)

    # 分割用到的关键点
    input_point = np.array([left_shoulder_point, left_elbow_point, left_wrist_point])
    input_label = np.array([1, 1, 1])

    # 分割用到的矩形框
    left_arm_x = []
    left_arm_x.append(left_shoulder_point[0])
    left_arm_x.append(left_elbow_point[0])
    left_arm_x.append(left_wrist_point[0])
    left_arm_x.sort()

    left_arm_y = []
    left_arm_y.append(left_shoulder_point[1])
    left_arm_y.append(left_elbow_point[1])
    left_arm_y.append(left_wrist_point[1])
    left_arm_y.sort()

    left_arm_box = []
    left_arm_box.append(left_arm_x[0]-40)
    left_arm_box.append(left_arm_y[0]-10)
    left_arm_box.append(left_arm_x[2]+40)
    left_arm_box.append(left_arm_y[2])

    input_box = []
    input_box.append(left_arm_box)
    input_box = np.array(input_box)

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        box=input_box[None, :],
        multimask_output=True,
    )

    mask_input = logits[np.argmax(scores), :, :]

    masks, _, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        box=input_box[None, :],
        mask_input=mask_input[None, :, :],
        multimask_output=False,
    )

    # 确保 masks 是二维的
    if len(masks.shape) == 3 and masks.shape[0] == 1:  # 如果 masks 形状为 [1, H, W]
        masks = masks[0]  # 取出第一个通道

    # 绘制掩码
    # 创建一个与原图同尺寸的空白彩色画布
    colored_mask = np.zeros_like(image)
    # 填充选中和未选中的掩码区域
    colored_mask[masks > 0] = (255, 0, 0)  # 红色
    colored_mask[masks == 0] = (124, 252, 0)  # 绿色

    # 在原图上绘制掩码
    canvas = np.zeros_like(image)  # 创建一个空画布，用于绘制
    canvas = image.copy()  # 在画布上绘制原始图像
    canvas = cv2.addWeighted(canvas, 0.7, colored_mask, 0.3, 0)  # 调整原图与掩码的透明度比例并叠加

    # 是否绘制关键点
    # for point, label in zip(input_point, input_label):
    #     color = (0, 0, 0) if label == 1 else (0, 0, 0)  # 正向点绿色，反向点红色
    #     canvas = cv2.circle(canvas, (int(point[0]), int(point[1])), radius=5, color=color, thickness=-1)
    #     colored_mask = cv2.circle(colored_mask, (int(point[0]), int(point[1])), radius=5, color=color, thickness=-1)
    
    # 绘制 box 框
    # 获取矩形框的左上角和右下角坐标
    # box_top_left = (int(left_arm_box[0]), int(left_arm_box[1]))
    # box_bottom_right = (int(left_arm_box[2]), int(left_arm_box[3]))
    # # 在 canvas 上绘制矩形框
    # canvas = cv2.rectangle(canvas, box_top_left, box_bottom_right, color=(0, 0, 0), thickness=2)  # 黑色框
    # colored_mask = cv2.rectangle(colored_mask, box_top_left, box_bottom_right, color=(0, 0, 0), thickness=2)  # 黑色框


    image_name = os.path.basename(image_path)[:-4]
    output_image_path = os.path.join(output_dir, f"{image_name}.jpg")

    # 只保存掩码图像
    colored_mask = cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR)  # 将图像转回 OpenCV 格式
    cv2.imwrite(output_image_path, colored_mask)

    # 保存原图与掩码的叠加图
    # canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)  # 将图像转回 OpenCV 格式
    # cv2.imwrite(output_image_path, canvas)

    # 返回分割后的图像路径
    return output_image_path

def seg_person_right_arm(image_path, json_path, output_dir, sam):
    # 对于衣服的分割，采用点选 + 框选的方法
    # 点选：右胳膊（右肩，右肘，右腕）
    # 框选：根据胳膊关键点去放大坐标框选胳膊

    from segment_anything import sam_model_registry, SamPredictor

    # 获取关键点
    ups_points = get_points(json_path)
    right_shoulder_point = ups_points[9]
    right_elbow_point = ups_points[10]
    right_wrist_point = ups_points[11]

    # 打开图片
    print(image_path)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 加载模型处理关键点
    # sam_checkpoint = "/home/xkmb/tryondiffusion/segment_anything_main/checkpoints/sam_vit_h_4b8939.pth"
    # sam_hq_checkpoint = "/home/xkmb/tryondiffusion/segment_anything_main/checkpoints/sam_hq_vit_h.pth"
    # model_type = "vit_h"
    # sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    # sam.to(device="cuda")
    predictor = SamPredictor(sam)
    predictor.set_image(image)

    # 分割用到的关键点
    input_point = np.array([right_shoulder_point, right_elbow_point, right_wrist_point])
    input_label = np.array([1, 1, 1])

    # 分割用到的矩形框
    right_arm_x = []
    right_arm_x.append(right_shoulder_point[0])
    right_arm_x.append(right_elbow_point[0])
    right_arm_x.append(right_wrist_point[0])
    right_arm_x.sort()

    right_arm_y = []
    right_arm_y.append(right_shoulder_point[1])
    right_arm_y.append(right_elbow_point[1])
    right_arm_y.append(right_wrist_point[1])
    right_arm_y.sort()

    right_arm_box = []
    right_arm_box.append(right_arm_x[0]-40)
    right_arm_box.append(right_arm_y[0]-10)
    right_arm_box.append(right_arm_x[2]+40)
    right_arm_box.append(right_arm_y[2])

    input_box = []
    input_box.append(right_arm_box)
    input_box = np.array(input_box)

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        box=input_box[None, :],
        multimask_output=True,
    )

    mask_input = logits[np.argmax(scores), :, :]

    masks, _, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        box=input_box[None, :],
        mask_input=mask_input[None, :, :],
        multimask_output=False,
    )

    # 确保 masks 是二维的
    if len(masks.shape) == 3 and masks.shape[0] == 1:  # 如果 masks 形状为 [1, H, W]
        masks = masks[0]  # 取出第一个通道

    # 绘制掩码
    # 创建一个与原图同尺寸的空白彩色画布
    colored_mask = np.zeros_like(image)
    # 填充选中和未选中的掩码区域
    colored_mask[masks > 0] = (255, 0, 0)  # 红色
    colored_mask[masks == 0] = (124, 252, 0)  # 绿色

    # 在原图上绘制掩码
    canvas = np.zeros_like(image)  # 创建一个空画布，用于绘制
    canvas = image.copy()  # 在画布上绘制原始图像
    canvas = cv2.addWeighted(canvas, 0.7, colored_mask, 0.3, 0)  # 调整原图与掩码的透明度比例并叠加

    # 是否绘制关键点
    # for point, label in zip(input_point, input_label):
    #     color = (0, 0, 0) if label == 1 else (0, 0, 0)  # 正向点绿色，反向点红色
    #     canvas = cv2.circle(canvas, (int(point[0]), int(point[1])), radius=5, color=color, thickness=-1)
    #     colored_mask = cv2.circle(colored_mask, (int(point[0]), int(point[1])), radius=5, color=color, thickness=-1)
    
    # # 绘制 box 框
    # # 获取矩形框的左上角和右下角坐标
    # box_top_left = (int(right_arm_box[0]), int(right_arm_box[1]))
    # box_bottom_right = (int(right_arm_box[2]), int(right_arm_box[3]))
    # # 在 canvas 上绘制矩形框
    # canvas = cv2.rectangle(canvas, box_top_left, box_bottom_right, color=(0, 0, 0), thickness=2)  # 黑色框
    # colored_mask = cv2.rectangle(colored_mask, box_top_left, box_bottom_right, color=(0, 0, 0), thickness=2)  # 黑色框

    image_name = os.path.basename(image_path)[:-4]
    output_image_path = os.path.join(output_dir, f"{image_name}.jpg")

    # 只保存掩码图像
    colored_mask = cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR)  # 将图像转回 OpenCV 格式
    cv2.imwrite(output_image_path, colored_mask)

    # 保存原图与掩码的叠加图
    # canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)  # 将图像转回 OpenCV 格式
    # cv2.imwrite(output_image_path, canvas)

    # 返回分割后的图像路径
    return output_image_path

def seg_person_neck(image_path, json_path, output_dir, sam):
    # 对于衣服的分割，采用点选 + 框选的方法
    # 点选：计算出脖子正中心
    # 框选：根据肩膀 + 下巴点构成选择框，框选住脖子
    # 效果不好，暂时不使用这个

    from segment_anything import sam_model_registry, SamPredictor

    # 获取关键点
    ups_points = get_points(json_path)
    nose_point = ups_points[0]
    left_shoulder_point = ups_points[6]
    right_shoulder_point = ups_points[9]
    left_chest_point = ups_points[3]
    right_chest_point = ups_points[4]
    # 计算得到左右两肩膀点的中间点
    mid_shoulder_point = []
    mid_shoulder_point_x = []
    mid_shoulder_point_x.append(left_shoulder_point[0])
    mid_shoulder_point_x.append(right_shoulder_point[0])
    mid_shoulder_point_x.sort()
    mid_shoulder_point_y = []
    mid_shoulder_point_y.append(left_shoulder_point[1])
    mid_shoulder_point_y.append(right_shoulder_point[1])
    mid_shoulder_point_y.sort()
    mid_shoulder_point.append(mid_shoulder_point_x[0]+(mid_shoulder_point_x[1]-mid_shoulder_point_x[0])/2)
    mid_shoulder_point.append(mid_shoulder_point_y[0]+(mid_shoulder_point_y[1]-mid_shoulder_point_y[0])/2)
    # 计算得到脖子关键点：鼻子与肩膀中点连线的1/3处
    nose_mid_shoulder_x = []
    nose_mid_shoulder_x.append(nose_point[0])
    nose_mid_shoulder_x.append(mid_shoulder_point[0])
    nose_mid_shoulder_x.sort()
    nose_mid_shoulder_y = []
    nose_mid_shoulder_y.append(nose_point[1])
    nose_mid_shoulder_y.append(mid_shoulder_point[1])
    nose_mid_shoulder_y.sort()

    if (nose_point[0] <= mid_shoulder_point[0]):  # 头右斜
        neck_point = []
        neck_point.append(nose_mid_shoulder_x[0]+3*(nose_mid_shoulder_x[1]-nose_mid_shoulder_x[0])/5)
        neck_point.append(nose_mid_shoulder_y[0]+(nose_mid_shoulder_y[1]-nose_mid_shoulder_y[0])/2)
        jaw_point = []
        jaw_point.append(nose_mid_shoulder_x[0]+(nose_mid_shoulder_x[1]-nose_mid_shoulder_x[0])/3)
        jaw_point.append(nose_mid_shoulder_y[0]+(nose_mid_shoulder_y[1]-nose_mid_shoulder_y[0])/3)
    else:  # 头左斜
        neck_point = []
        neck_point.append(nose_mid_shoulder_x[0]+2*(nose_mid_shoulder_x[1]-nose_mid_shoulder_x[0])/5)
        neck_point.append(nose_mid_shoulder_y[0]+(nose_mid_shoulder_y[1]-nose_mid_shoulder_y[0])/2)
        jaw_point = []
        jaw_point.append(nose_mid_shoulder_x[0]+2*(nose_mid_shoulder_x[1]-nose_mid_shoulder_x[0])/3)
        jaw_point.append(nose_mid_shoulder_y[0]+(nose_mid_shoulder_y[1]-nose_mid_shoulder_y[0])/3)

    # 打开图片
    print(image_path)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 加载模型处理关键点
    # sam_checkpoint = "/home/xkmb/tryondiffusion/segment_anything_main/checkpoints/sam_vit_h_4b8939.pth"
    # sam_hq_checkpoint = "/home/xkmb/tryondiffusion/segment_anything_main/checkpoints/sam_hq_vit_h.pth"
    # model_type = "vit_h"
    # sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    # sam.to(device="cuda")
    predictor = SamPredictor(sam)
    predictor.set_image(image)

    # 分割用到的关键点
    input_point = np.array([neck_point])
    input_label = np.array([1])

    # 分割用到的矩形框
    neck_x = []
    neck_x.append(left_shoulder_point[0])
    neck_x.append(right_shoulder_point[0])
    neck_x.append(jaw_point[0])
    neck_x.append(left_shoulder_point[0])
    neck_x.append(right_shoulder_point[0])
    # neck_x.append(left_chest_point[0])
    # neck_x.append(right_chest_point[0])
    neck_x.sort()

    neck_y = []
    neck_y.append(left_shoulder_point[1])
    neck_y.append(right_shoulder_point[1])
    neck_y.append(jaw_point[1])
    neck_y.append(left_shoulder_point[1])
    neck_y.append(right_shoulder_point[1])
    # neck_y.append(left_chest_point[1])
    # neck_y.append(right_chest_point[1])
    neck_y.sort()

    neck_box = []
    neck_box.append(neck_x[0])
    neck_box.append(neck_y[0])
    neck_box.append(neck_x[4])
    neck_box.append(neck_y[4])

    input_box = []
    input_box.append(neck_box)
    input_box = np.array(input_box)

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        box=input_box[None, :],
        multimask_output=True,
    )

    mask_input = logits[np.argmax(scores), :, :]

    masks, _, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        box=input_box[None, :],
        mask_input=mask_input[None, :, :],
        multimask_output=False,
    )

    # 确保 masks 是二维的
    if len(masks.shape) == 3 and masks.shape[0] == 1:  # 如果 masks 形状为 [1, H, W]
        masks = masks[0]  # 取出第一个通道

    # 绘制掩码
    # 创建一个与原图同尺寸的空白彩色画布
    colored_mask = np.zeros_like(image)
    # 填充选中和未选中的掩码区域
    colored_mask[masks > 0] = (255, 0, 0)  # 红色
    colored_mask[masks == 0] = (124, 252, 0)  # 绿色

    # 在原图上绘制掩码
    canvas = np.zeros_like(image)  # 创建一个空画布，用于绘制
    canvas = image.copy()  # 在画布上绘制原始图像
    canvas = cv2.addWeighted(canvas, 0.7, colored_mask, 0.3, 0)  # 调整原图与掩码的透明度比例并叠加

    # 是否绘制关键点
    # for point, label in zip(input_point, input_label):
    #     color = (0, 0, 0) if label == 1 else (0, 0, 0)  # 正向点绿色，反向点红色
    #     canvas = cv2.circle(canvas, (int(point[0]), int(point[1])), radius=5, color=color, thickness=-1)
    #     colored_mask = cv2.circle(colored_mask, (int(point[0]), int(point[1])), radius=5, color=color, thickness=-1)
    
    # # 绘制 box 框
    # # 获取矩形框的左上角和右下角坐标
    # box_top_left = (int(neck_box[0]), int(neck_box[1]))
    # box_bottom_right = (int(neck_box[2]), int(neck_box[3]))
    # # 在 canvas 上绘制矩形框
    # canvas = cv2.rectangle(canvas, box_top_left, box_bottom_right, color=(0, 0, 0), thickness=2)  # 黑色框
    # colored_mask = cv2.rectangle(colored_mask, box_top_left, box_bottom_right, color=(0, 0, 0), thickness=2)  # 黑色框

    image_name = os.path.basename(image_path)[:-4]
    output_image_path = os.path.join(output_dir, f"{image_name}.jpg")

    # 只保存掩码图像
    colored_mask = cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR)  # 将图像转回 OpenCV 格式
    cv2.imwrite(output_image_path, colored_mask)

    # 保存原图与掩码的叠加图
    # canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)  # 将图像转回 OpenCV 格式
    # cv2.imwrite(output_image_path, canvas)

    # 返回分割后的图像路径
    return output_image_path

def seg_any(image_dir, json_dir, output_dir, seg_something, sam):
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取 image_dir 中的所有文件名
    image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]

    # 对于每个图像文件，找到对应的 JSON 文件
    for image_file in image_files:
        # 构建图像文件的完整路径
        image_path = os.path.join(image_dir, image_file)

        # 构建相应的 JSON 文件名
        json_file = os.path.splitext(image_file)[0] + '.json'
        json_path = os.path.join(json_dir, json_file)

        # 检查 JSON 文件是否存在
        if os.path.exists(json_path):
            seg_something(image_path, json_path, output_dir, sam)
        else:
            print(f"未找到 JSON 文件: {json_path}")

def seg_anyone(an_image_path, a_json_path, output_dir, seg_something, sam):
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 检查 JSON 文件是否存在
    if os.path.exists(a_json_path):
        return seg_something(an_image_path, a_json_path, output_dir, sam)
    else:
        print(f"未找到 JSON 文件: {a_json_path}")

if __name__ == "__main__":
    # 加载模型
    sam_checkpoint = "/home/xkmb/tryondiffusion/models/sam_vit_h_4b8939.pth"
    sam_hq_checkpoint = "/home/xkmb/tryondiffusion/models/sam_hq_vit_h.pth"
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device="cuda")


    # image_dir = '/home/xkmb/data/train/ig'
    # json_dir = '/home/xkmb/pose_output/train/ig/predictions'
    # mask_dir = '/home/xkmb/data/train/ig_mask'

    # seg_any(image_dir, json_dir, mask_dir, seg_up, sam)