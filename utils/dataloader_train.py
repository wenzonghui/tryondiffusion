import os
import cv2
import torch
import json
import math
import numbers
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
from torchvision import transforms as T

from utils.utils import load_pose_embed

def write_img(img, folder_path, img_name):
    path = os.path.join(folder_path, img_name)
    cv2.imwrite(path, img)


class GaussianSmoothing(nn.Module):
    """
    Source: https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/10?u=tanay_agrawal
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """

    def __init__(self, channels, kernel_size, sigma, conv_dim):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * conv_dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * conv_dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if conv_dim == 1:
            self.conv = F.conv1d
        elif conv_dim == 2:
            self.conv = F.conv2d
        elif conv_dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(
                    conv_dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)


def mk_folders(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name, "images"), exist_ok=True)


class ToPaddedTensorImages:
    def __call__(self, image):
        """Padding image so that aspect ratio is maintained.
        And converting numpy arrays to tensors."""
        # cv2 image: H x W x C
        # torch image: C X H X W

        img = image.transpose((2, 0, 1))
        img = torch.from_numpy(img.copy()).float()

        if img.shape[1] > img.shape[2]:
            pad_size = (img.shape[1] - img.shape[2]) // 2
            padding = (pad_size, pad_size, 0, 0)
        elif img.shape[2] > img.shape[1]:
            pad_size = (img.shape[2] - img.shape[1]) // 2
            padding = (0, 0, pad_size, pad_size)
        else:
            padding = (0, 0, 0, 0)

        img = F.pad(img, padding, "constant", 0)

        return img


class ToTensorEmbed:
    def __call__(self, pose_embed):
        return torch.from_numpy(pose_embed)


def create_transforms_imgs(image, unet_size):
    transforms = T.Compose([
        ToPaddedTensorImages(),  # 假设这个函数可以直接应用于图像
        T.Resize(unet_size),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    return transforms(image)


class UNetDataset(Dataset):
    """ This class is to be used while training, where all the conditional inputs and ground
     truth is pre-saved and are pre-processed."""

    def __init__(self, ip_dir, jp_dir, jg_dir, ia_dir, ic_dir, itr128_dir, unet_size):
        """
        Get all the inputs from ../data directory in the main project directory
        :param ip_dir: Image of target person with source clothing on. Later
        to be used to generate zt and to be used as ground truth for training.
        :param jp_dir: person pose embeddings from ip
        :param jg_dir: garment pose embeddings from 'ig', ig is the source garment image
        :param ia_dir: clothing agnostic rgb from ip
        :param ic_dir: segmented garment from ig
        """
        self.ip_list = os.listdir(ip_dir)
        self.ip_paths = [os.path.join(ip_dir, i) for i in self.ip_list]

        self.jp_list = os.listdir(jp_dir)
        self.jp_paths = [os.path.join(jp_dir, i) for i in self.jp_list]

        self.jg_list = os.listdir(jg_dir)
        self.jg_paths = [os.path.join(jg_dir, i) for i in self.jg_list]

        self.ia_list = os.listdir(ia_dir)
        self.ia_paths = [os.path.join(ia_dir, i) for i in self.ia_list]

        self.ic_list = os.listdir(ic_dir)
        self.ic_paths = [os.path.join(ic_dir, i) for i in self.ic_list]

        self.itr128_list = os.listdir(itr128_dir)
        self.itr128_paths = [os.path.join(itr128_dir, i)
                             for i in self.itr128_list]

        self.transforms_imgs = T.Compose([
            ToPaddedTensorImages(),
            T.Resize(unet_size),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __len__(self):
        return len(self.ip_list)

    # 原本的写法，返回的直接是对应的元素内容
    # def __getitem__(self, item):
    #     jp = load_pose_embed((self.jp_paths[item]))  # 这种是直接得到了 jp 中的数据
    #     jg = load_pose_embed((self.jg_paths[item]))

    #     ip = read_img(self.ip_paths[item])
    #     ia = read_img(self.ia_paths[item])
    #     ic = read_img(self.ic_paths[item])
    #     itr128 = read_img(self.itr128_paths[item])

    #     ip = self.transforms_imgs(ip)  # 在后面再处理
    #     ia = self.transforms_imgs(ia) # 在后面再处理
    #     ic = self.transforms_imgs(ic) # 在后面再处理
    #     itr128 = self.transforms_imgs(itr128) # 在后面再处理

    # 这种写法只返回对应的文件路径
    def __getitem__(self, item):
        jp = self.jp_paths[item]  # 这种只得到了 jp 的文件路径，数据在后面再进行读取
        jg = self.jg_paths[item]

        ip = self.ip_paths[item]
        ia = self.ia_paths[item]
        ic = self.ic_paths[item]
        itr128 = self.itr128_paths[item]

        # 经过这样处理，返回的全部是文件路径
        return ip, jp, jg, ia, ic, itr128
