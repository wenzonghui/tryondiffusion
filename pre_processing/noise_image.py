# 这个文件暂时没有用到

from torchvision.transforms.v2 import GaussianBlur, PILToTensor


def add_gaussian_blur(img):
    """
    Add Gaussian Blur
    :param img: input image
    :return: noisy image
    """
    gb = GaussianBlur(kernel_size=(3, 3), sigma=(0.2, 0.6))
    img = PILToTensor()(img)
    return gb(img)


if __name__ == "__main__":
    import os
    import sys

    parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, parentdir)

    from utils.utils import read_img, write_img

    img = read_img("data/ip/00000_00.jpg")

    segmented_garment = add_gaussian_blur(img)

    write_img(segmented_garment, "./data/zt", "00000_00.jpg")
