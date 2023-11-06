import numpy as np


def get_upper_garment(img, img_parse_map):
    sum_img_parse_map = np.sum(img_parse_map, axis=2)
    sum_img_parse_map[sum_img_parse_map!=339] = 0
    sum_img_parse_map[sum_img_parse_map==339] = 1
    upper_garment_segment = (sum_img_parse_map.reshape(*sum_img_parse_map.shape,1)*img).astype(np.uint8)
    return upper_garment_segment


if __name__ == "__main__":
    import os
    import sys

    parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, parentdir)

    from utils.utils import read_img, write_img

    img = read_img("./data/garment/00006_00.jpg")
    img_parse_map = read_img("./data/image-parse-v3/00006_00.png")

    segmented_garment = get_upper_garment(img, img_parse_map)

    write_img(segmented_garment, "./data/ic", "00006_00.jpg")
