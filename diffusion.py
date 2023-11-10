import copy
import json
import logging
import os
import time

import torch
from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn
from torch.nn import functional as F
import cv2
import numpy as np
from UNet128 import UNet128
from UNet256 import UNet256
from utils.utils import mk_folders, GaussianSmoothing
from utils.dataloader_train import UNetDataset
from ema import EMA
from pre_processing.person_pose_embedding.network import AutoEncoder as PersonAutoEncoder
from pre_processing.garment_pose_embedding.network import AutoEncoder as GarmentAutoEncoder


def smoothen_image(img, sigma):
    # As suggested in: https://jmlr.csail.mit.edu/papers/volume23/21-0635/21-0635.pdf Section 4.4
    # 高斯噪音增强函数

    smoothing2d = GaussianSmoothing(channels=3,
                                    kernel_size=3,
                                    sigma=sigma,
                                    conv_dim=2)

    img = F.pad(img, (1, 1, 1, 1), mode='reflect')
    img = smoothing2d(img)

    return img


# 学习计划率、根据训练步骤调整学习率
def schedule_lr(total_steps, start_lr=0.0, stop_lr=0.0001, pct_increasing_lr=0.02):
    n = total_steps * pct_increasing_lr
    n = round(n)
    lambdas = list(np.linspace(start_lr, stop_lr, n))
    constant_lr_list = [stop_lr] * (total_steps - n)
    lambdas.extend(constant_lr_list)

    return lambdas


class Diffusion:

    def __init__(self,
                 device,
                 pose_embed_dim,
                 time_steps=256,
                 beta_start=1e-4,
                 beta_end=0.02,
                 unet_dim=128,
                 noise_input_channel=3,
                 beta_ema=0.995):
        self.time_steps = time_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta = self.linear_beta_scheduler().to(device)
        self.alpha = 1 - self.beta
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0)
        self.noise_input_channel = noise_input_channel
        self.unet_dim = unet_dim
        self.beta_ema = beta_ema
        self.device = device

        if unet_dim == 128:
            self.net = UNet128(pose_embed_dim, device, time_steps).to(device)
        elif unet_dim == 256:
            self.net = UNet256(pose_embed_dim, device, time_steps).to(device)

        self.fc1 = PersonAutoEncoder(50)
        self.fc1.load_state_dict(
            torch.load('/home/xkmb/tryondiffusion/data/train/jp_embed/best_model.pth', map_location=self.device))
        self.fc2 = GarmentAutoEncoder(20)
        self.fc2.load_state_dict(
            torch.load('/home/xkmb/tryondiffusion/data/train/jg_embed/best_model.pth', map_location=self.device))

        self.ema_net = copy.deepcopy(self.net).eval().requires_grad_(False)

    # 产生一个扩散过程的线性 beta 调度
    def linear_beta_scheduler(self):
        return torch.linspace(self.beta_start, self.beta_end, self.time_steps)

    # 随机采样时间步骤
    def sample_time_steps(self, batch_size):
        return torch.randint(low=1, high=self.time_steps, size=(batch_size,))

    # 这里是匹配论文中的 Noisy image，也就是通过 Ip 生成 Zt
    def add_noise_to_img(self, img, t):
        sqrt_alpha_timestep = torch.sqrt(self.alpha_cumprod[t])[:, None, None, None]
        sqrt_one_minus_alpha_timestep = torch.sqrt(1 - self.alpha_cumprod[t])[:, None, None, None]
        epsilon = torch.randn_like(img)
        return (sqrt_alpha_timestep * epsilon) + (sqrt_one_minus_alpha_timestep * epsilon), epsilon

    @torch.inference_mode()
    def sample(self, use_ema, conditional_inputs):
        model = self.ema_net if use_ema else self.net
        ic, jp, jg, ia = conditional_inputs
        ic = ic.to(self.device)
        jp = jp.to(self.device)
        jg = jg.to(self.device)
        ia = ia.to(self.device)
        batch_size = len(ic)
        logging.info(f"Running inference for {batch_size} images")

        model.eval()
        with torch.inference_mode():

            # noise augmentation during testing as suggested in paper
            sigma = float(torch.FloatTensor(1).uniform_(0.4, 0.6))
            ia = smoothen_image(ia, sigma)
            ic = smoothen_image(ic, sigma)

            inp_network_noise = torch.randn(batch_size, self.noise_input_channel, self.unet_dim, self.unet_dim).to(
                self.device)

            # paper says to add noise augmentation to input noise during inference
            inp_network_noise = smoothen_image(inp_network_noise, sigma)

            # concatenating noise with rgb agnostic image across channels
            # corrupt -> concatenate -> predict
            x = torch.cat((inp_network_noise, ia), dim=1)

            for i in reversed(range(1, self.time_steps)):
                t = (torch.ones(batch_size) * i).long().to(self.device)
                predicted_noise = model(x, ic, jp, jg, t, sigma)

                # ToDo: Add Classifier-Free Guidance with guidance weight 2
                alpha = self.alpha[t][:, None, None, None]
                alpha_cumprod = self.alpha_cumprod[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]

                if i > 1:
                    noise = torch.randn_like(inp_network_noise)
                else:
                    noise = torch.zeros_like(inp_network_noise)

                inp_network_noise = 1 / torch.sqrt(alpha) * (inp_network_noise - (
                        (1 - alpha) / (torch.sqrt(1 - alpha_cumprod))) * predicted_noise) + torch.sqrt(beta) * noise
        inp_network_noise = (inp_network_noise.clamp(-1, 1) + 1) / 2
        inp_network_noise = (inp_network_noise * 255).type(torch.uint8)

        return inp_network_noise

    # 准备训练所需要的数据加载器、优化器、EMA指数移动平均
    def prepare(self, args):
        mk_folders(args.run_name)
        train_dataset = UNetDataset(ip_dir=args.train_ip_folder,
                                    jp_dir=args.train_jp_folder,
                                    jg_dir=args.train_jg_folder,
                                    ia_dir=args.train_ia_folder,
                                    ic_dir=args.train_ic_folder,
                                    unet_size=self.unet_dim)

        validation_dataset = UNetDataset(ip_dir=args.validation_ip_folder,
                                         jp_dir=args.validation_jp_folder,
                                         jg_dir=args.validation_jg_folder,
                                         ia_dir=args.validation_ia_folder,
                                         ic_dir=args.validation_ic_folder,
                                         unet_size=self.unet_dim)

        self.train_dataloader = DataLoader(train_dataset, args.batch_size_train, shuffle=True)

        # give args.batch_size_validation 1 while training
        self.val_dataloader = DataLoader(validation_dataset, args.batch_size_validation, shuffle=True)
        self.optimizer = optim.AdamW(self.net.parameters(), lr=args.lr, eps=1e-4)
        self.scheduler = schedule_lr(total_steps=round((args.data_len * args.epochs) / args.batch_size_train),
                                     start_lr=args.start_lr, stop_lr=args.stop_lr,
                                     pct_increasing_lr=args.pct_increasing_lr)
        self.mse = nn.MSELoss()
        self.ema = EMA(self.beta_ema)
        self.scaler = torch.cuda.amp.GradScaler()

    # 进行单个训练步骤，包括反向传播和参数更新
    def train_step(self, loss, running_step):
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.ema.step_ema(self.ema_net, self.net)

        for g in self.optimizer.param_groups:
            g['lr'] = self.scheduler[running_step]

    # 处理一个训练/验证周期，并可以选择性的打印出损失
    def single_epoch(self, epoch=-1, epochs=-1, every_epoch_steps=-1, train=True):
        avg_loss = 0.

        if train:
            self.net.train()

        else:
            self.net.eval()

        for ip, jp, jg, ia, ic in self.train_dataloader:

            # 训练过程中打印步数
            print("Epoch: " + str(epoch) + "/" + str(epochs) + " ::: " + "Step: " + str(
                self.running_train_steps) + "/" + str(every_epoch_steps))
            # self.save_models(0)

            # 这里是针对每个 epoch 过大，在中间 10% 进行一步模型权重存储临时使用
            if self.running_train_steps == round(0.01 * every_epoch_steps):
                print("Now save checkpoints.")
                self.save_models(0)

            # noise augmentation
            # 在任何其他处理之前，向 ia、ic 添加随机高斯噪声进行噪声增强
            sigma = float(torch.FloatTensor(1).uniform_(0.4, 0.6))
            ia = smoothen_image(ia, sigma)
            ic = smoothen_image(ic, sigma)

            with torch.autocast(self.device) and (torch.inference_mode() if not train else torch.enable_grad()):
                ip = ip.to(self.device)

                # 这里得到的是一个 jp json 的文件路径，先读取 json 的内容，转换成 tensor，再通过 FC 网络处理。到这里都是在 CUDA1上进行的
                # 最后把处理后的 embedding jp 放在 CUDA0 上
                jp_data = []
                for jp_item in jp:
                    with open(jp_item, 'r') as jp_item:
                        jp_json = json.load(jp_item)
                        jp_data.append(jp_json)
                jp_tensor = torch.tensor(jp_data)
                jp_fc1 = self.fc1(jp_tensor)
                # jp = torch.tensor(jp_fc1[1]).to(self.device) # 换下面这种写法
                jp = jp_fc1[1].clone().detach().to(self.device)
                # jp = jp.cpu().to(self.device)

                jg_data = []
                for jg_item in jg:
                    with open(jg_item, 'r') as jg_item:
                        jg_json = json.load(jg_item)
                        jg_data.append(jg_json)
                jg_tensor = torch.tensor(jg_data)
                jg_fc2 = self.fc2(jg_tensor)
                # jg = torch.tensor(jg_fc2[1]).to(self.device) # 换下面这种写法
                jg = jg_fc2[1].clone().detach().to(self.device)
                # jg = jg.cpu().to(self.device)

                ia = ia.to(self.device)
                ic = ic.to(self.device)
                t = self.sample_time_steps(ip.shape[0]).to(self.device)

                # corrupt -> concatenate -> predict
                # 对 ip 添加 noise 变成 zt
                zt, noise_epsilon = self.add_noise_to_img(ip, t)

                # ToDO: 这里只针对了 UNet128，并未把 UNet128 的结果 Itr128 一起 concat 进去，以输入 UNet256 网络进行训练
                # zt 与 ia 进行 concat，用 zt 表示，准备将数据输入网络中
                zt = torch.cat((zt, ia), dim=1).to(self.device)

                # ToDO: Make conditional inputs null, at 10% of the training time, for classifier-free guidance(GitHub Issue #21), with guidance weight 2.

                predicted_noise = self.net(zt, ic, jp, jg, t, sigma)
                loss = self.mse(noise_epsilon, predicted_noise)
                avg_loss += loss

            if train:
                self.train_step(loss, self.running_train_steps)
                # ToDo: Add logs to tensorboard as well
                logging.info(
                    f"train_mse_loss: {loss.item():2.3f}, learning_rate: {self.scheduler[self.running_train_steps]}")
                self.running_train_steps += 1

        return avg_loss.mean().item()

    # 在训练中记录图像样本
    def logging_images(self, epoch, run_name):

        for idx, (ip, jp, jg, ia, ic) in enumerate(self.val_dataloader):
            # sampled image
            sampled_image = self.sample(use_ema=False, conditional_inputs=(ic, jp, jg, ia))
            sampled_image = sampled_image[0].permute(1, 2, 0).squeeze().cpu().numpy()

            # ema sampled image
            ema_sampled_image = self.sample(use_ema=True, conditional_inputs=(ic, jp, jg, ia))
            ema_sampled_image = ema_sampled_image[0].permute(1, 2, 0).squeeze().cpu().numpy()

            # base images
            ip_np = ip[0].permute(1, 2, 0).squeeze().cpu().numpy()
            ic_np = ic[0].permute(1, 2, 0).squeeze().cpu().numpy()
            ia_np = ia[0].permute(1, 2, 0).squeeze().cpu().numpy()

            # make to folders
            os.makedirs(os.path.join("results", run_name, "images", f"{idx}_E{epoch}"), exist_ok=True)

            # define folder paths
            images_folder = os.path.join("results", run_name, "images", f"{idx}_E{epoch}")

            # save base images
            cv2.imwrite(os.path.join(images_folder, "ground_truth.png"), ip_np)
            cv2.imwrite(os.path.join(images_folder, "segmented_garment.png"), ic_np)
            cv2.imwrite(os.path.join(images_folder, "cloth_agnostic_rgb.png"), ia_np)

            # save sampled image
            cv2.imwrite(os.path.join(images_folder, "sampled_image.png"), sampled_image)

            # save ema sampled image
            cv2.imwrite(os.path.join(images_folder, "ema_sampled_image.png"), ema_sampled_image)

    # 保存模型的权重和优化器状态
    def save_models(self, epoch=-1):
        torch.save(self.net.state_dict(), os.path.join("models", "ckpt128", f"ckpt_{epoch}.pt"))
        torch.save(self.ema_net.state_dict(), os.path.join("models", "ema_ckpt128", f"ema_ckpt_{epoch}.pt"))
        torch.save(self.optimizer.state_dict(), os.path.join("models", "optim128", f"optim_{epoch}.pt"))

    # 开始和管理训练过程
    def fit(self, args):

        logging.info(f"Starting training")

        print("args.total_steps: ", round((args.data_len * args.epochs) / args.batch_size_train))
        print("args.batch_size_train: ", args.batch_size_train)
        print("data_len: ", args.data_len)
        print("\nEpochs: ", args.epochs)

        if args.epochs < 0:
            args.epochs = 1

        self.running_train_steps = 0
        # 每一个 epoch 包含多少 steps
        every_epoch_steps = round(args.data_len / args.batch_size_train)
        print("Every epoch steps: ", every_epoch_steps)

        for epoch in range(args.epochs):
            print(f"\nStarting Epoch: {epoch + 1} / {args.epochs}")
            start_time = time.time()
            _ = self.single_epoch(epoch, args.epochs, every_epoch_steps, train=True)
            end_time = time.time()

            # Calculate the total execution time in seconds
            total_time = end_time - start_time

            # Convert the time into hours, minutes, and seconds
            hours = int(total_time // 3600)
            minutes = int((total_time % 3600) // 60)
            seconds = total_time % 60
            print(f"Epoch Execution Time: {hours} hours, {minutes} minutes, and {seconds:.2f} seconds")

            if (epoch + 1) % args.calculate_loss_frequency == 0:
                avg_loss = self.single_epoch(train=False)
                logging.info(f"Average Loss: {avg_loss}")

            if (epoch + 1) % args.image_logging_frequency == 0:
                self.logging_images(epoch, args.run_name)

            if (epoch + 1) % args.model_saving_frequency == 0:
                self.save_models(epoch)

        logging.info(f"Training Done Successfully!")
