import os


def pose_model_process(imgs_path, output_dir):
    # 加载 pose model，image-->pose

    from mmpose.apis import MMPoseInferencer

    # 使用模型别名创建推断器
    inferencer = MMPoseInferencer(
        device='cuda',
        # 全身检测配置
        # pose2d='configs/wholebody_2d_keypoint/topdown_heatmap/coco-wholebody/td-hm_hrnet-w48_dark-8xb32-210e_coco-wholebody-384x288.py',
        # pose2d_weights='../models/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth',

        # 肢体检测配置
        pose2d='/home/xkmb/tryondiffusion/mmpose_main/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrformer-base_8xb32-210e_coco-384x288.py',
        pose2d_weights='/home/xkmb/tryondiffusion/models/hrformer_base_coco_384x288-ecf0758d_20220316.pth',

        # 检测模型配置，看是否使用检测模型
        # det_model=f'mmpose/configs/yolox/yolox_l_8x8_300e_coco.py',
        # det_weights='modelsyolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth',

        det_cat_ids=[0],  # 指定'human'类的类别id
    )

    # MMPoseInferencer采用了惰性推断方法，在给定输入时创建一个预测生成器
    result_generator = inferencer(imgs_path, out_dir=output_dir)
    results = [result for result in result_generator]
    # print("\nresults: ", results)
    print(f"{imgs_path} pose 识别结束")

    # 如果是单张图片而不是文件夹路径，主要用在推理部分
    if imgs_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
        img_name = imgs_path.split('/')[-1].split('.')[0]
        pose_json_path = os.path.join(output_dir, "predictions", f'{img_name}_pose.json')
        # 返回姿态估计的 json 文件路径
        return pose_json_path


if __name__ == "__main__":
    imgs_path = '/home/xkmb/tryondiffusion/mmpose_main/data/00006_00.jpg'
    output_dir = '/home/xkmb/tryondiffusion/mmpose_main/pose_output'
    pose_model_process(imgs_path, output_dir)
