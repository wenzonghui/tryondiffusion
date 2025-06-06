"""
处理 mmpose 得到的姿态估计 json
使得它适配 person_pose_embedding 和 garment_pose_embedding 函数的输入
"""


def start_pose_json_process(raw_json, output_dir):
    # 单张处理 mmpose 得到的 姿势识别 json，主要在推理阶段
    # 返回 json 文件的路径

    import json
    import os

    out_listxy = []
    with open(raw_json, "r") as f:
        img_pose_json = json.load(f)
        points = img_pose_json[0]["keypoints"]
        for point in points:
            out_listxy.append(point[0])
            out_listxy.append(point[1])
    json_data = json.dumps(out_listxy)

    # 这里返回的是处理后的 pose json 所在的文件路径，区别于 mmpose 刚处理后的
    json_name = raw_json.split('/')[-1].split('.')[0]
    pose_json_path = os.path.join(output_dir, f'{json_name}_pose_normlize.json')
    # 返回姿态估计的 json 文件路径

    # 存到文件夹里
    with open(pose_json_path, "w") as file:
        file.write(json_data)

    return pose_json_path


def start_pose_jsons_process(input_dir, output_dir):
    # 批量处理 mmpose 得到的 姿势识别 json，主要在预处理阶段

    import json
    import os

    current_wait_process_ig_jsonlist = os.listdir(os.path.join(input_dir, "ig", "predictions"))
    current_wait_process_ip_jsonlist = os.listdir(os.path.join(input_dir, "ip", "predictions"))
    output_ig_jsondir = os.path.join(output_dir, "jg")
    output_ip_jsondir = os.path.join(output_dir, "jp")

    for ig in current_wait_process_ig_jsonlist:
        print("ig: ", ig)
        ig_out_listxy = []
        # 这里需要先手动创建 ig 文件夹，再执行本程序
        with open(os.path.join(input_dir, "ig", "predictions", ig), "r") as f:
            img_pose_json = json.load(f)
            points = img_pose_json[0]["keypoints"]
            for point in points:
                ig_out_listxy.append(point[0])
                ig_out_listxy.append(point[1])
        json_data = json.dumps(ig_out_listxy)
        # print("ig json_data: ", json_data)
        with open(os.path.join(output_ig_jsondir, ig), "w") as file:
            file.write(json_data)

    for ip in current_wait_process_ip_jsonlist:
        print("ip: ", ip)
        ip_out_listxy = []
        # 这里需要先手动创建 ip 文件夹，再执行本程序
        with open(os.path.join(input_dir, "ip", "predictions", ip), "r") as f:
            img_pose_json = json.load(f)
            points = img_pose_json[0]["keypoints"]
            for point in points:
                ip_out_listxy.append(point[0])
                ip_out_listxy.append(point[1])
        json_data = json.dumps(ip_out_listxy)
        # print("ip json_data: ", json_data)
        with open(os.path.join(output_ip_jsondir, ip), "w") as file:
            file.write(json_data)


if __name__ == "__main__":
    # 批量处理 mmpose 得到的 姿势识别 json，主要在预处理阶段
    input_dir = "/home/xkmb/下载/data/pose/train"
    output_dir = "/home/xkmb/下载/data/train"

    start_pose_jsons_process(input_dir, output_dir)

# rm 0006152.jpg 0006091.jpg 0009314.jpg 0009336.jpg 0011986.jpg 0012071.jpg 0012535.jpg 0010771.jpg 0010425.jpg 0010301.jpg 0009071.jpg 0009056.jpg 0009016.jpg 0008583.jpg 0006899.jpg 0004555.jpg

# rm 0006152.json 0006091.json 0009314.json 0009336.json 0011986.json 0012071.json 0012535.json 0010771.json 0010425.json 0010301.json 0009071.json 0009056.json 0009016.json 0008583.json 0006899.json 0004555.json

# rm 0002506.jpg 0005489.jpg 0005630.jpg 0005777.jpg 0005778.jpg 0005835.jpg

# rm 0002506.json 0005489.json 0005630.json 0005777.json 0005778.json 0005835.json