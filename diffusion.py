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
from pre_processing.garment_pose_embedding.utils.dataloader import normalize_lst
from utils.utils import mk_folders, GaussianSmoothing, read_img
from utils.dataloader_train import UNetDataset, create_transforms_imgs
from ema import EMA
from pre_processing.person_pose_embedding.network import AutoEncoder as PersonAutoEncoder
from pre_processing.garment_pose_embedding.network import AutoEncoder as GarmentAutoEncoder


def smoothen_image(img, sigma):
    # As suggested in: https://jmlr.csail.mit.edu/papers/volume23/21-0635/21-0635.pdf Section 4.4
    # 高斯噪音增强函数
    # 输入：
    # • img：输入图像
    # • sigma：高斯核的标准差

    smoothing2d = GaussianSmoothing(channels=3,
                                    kernel_size=3,
                                    sigma=sigma,
                                    conv_dim=2)
    smoothing2d = smoothing2d.to('cuda')

    img = F.pad(img, (1, 1, 1, 1), mode='reflect')
    img = smoothing2d(img)

    return img


def schedule_lr(total_steps, start_lr=0.0, stop_lr=0.0001, pct_increasing_lr=0.02):
    # 学习计划率、根据训练步骤调整学习率
    # 参数:
    # • total_steps: 总训练步数
    # • start_lr 和 stop_lr: 学习率的开始和结束值
    # • pct_increasing_lr: 增加学习率的百分比

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
                 unet_dim=128,  # 默认训练 unet128，trainer 里可传参
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
            # DataParallel：DP单机多卡并行训练 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            # gpus = [0, 1, 2, 3, 4, 5, 6, 7]  # 这里使用8卡进行训练
            # self.net = nn.DataParallel(self.net, device_ids=gpus, output_device=gpus[0])
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        elif unet_dim == 256:
            self.net = UNet256(pose_embed_dim, device, time_steps).to(device)
            # DataParallel：DP单机多卡并行训练 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            # gpus = [0, 1, 2, 3, 4, 5, 6, 7]  # 这里使用8卡进行训练
            # self.net = nn.DataParallel(self.net, device_ids=gpus, output_device=gpus[0])
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        self.fc1 = PersonAutoEncoder(34)
        self.fc1.load_state_dict(torch.load('/home/xkmb/tryondiffusion/models/fc1.pth', map_location=self.device))
        self.fc2 = GarmentAutoEncoder(34)
        self.fc2.load_state_dict(torch.load('/home/xkmb/tryondiffusion/models/fc2.pth', map_location=self.device))

        self.ema_net = copy.deepcopy(self.net).eval().requires_grad_(False)

    def linear_beta_scheduler(self):
        # 产生一个扩散过程的线性 beta 调度
        return torch.linspace(self.beta_start, self.beta_end, self.time_steps)

    def sample_time_steps(self, batch_size):
        # 随机采样时间步骤
        return torch.randint(low=1, high=self.time_steps, size=(batch_size,))

    def add_noise_to_img(self, img, t):
        sqrt_alpha_timestep = torch.sqrt(self.alpha_cumprod[t])[:, None, None, None]
        sqrt_one_minus_alpha_timestep = torch.sqrt(1 - self.alpha_cumprod[t])[:, None, None, None]
        epsilon = torch.randn_like(img)
        return (sqrt_alpha_timestep * epsilon) + (sqrt_one_minus_alpha_timestep * epsilon), epsilon

    @torch.inference_mode()
    def sample(self, use_ema, conditional_inputs):
        # 生成样本图像
        # 参数：
        # • use_ema: 布尔值，指示是否使用EMA（指数移动平均）模型来生成图像。
        # • conditional_inputs: 条件输入，包括人物姿势、服装姿势和其他相关信息

        model = self.ema_net if use_ema else self.net
        ia, ic, jp, jg = conditional_inputs

        ia = ia.to(self.device)
        ic = ic.to(self.device)
        jp = jp.to(self.device)
        jg = jg.to(self.device)
        batch_size = len(ic)
        # print(f"Running inference for {batch_size} images")

        model.eval()
        with torch.inference_mode():
            # noise augmentation during testing as suggested in paper
            sigma = float(torch.FloatTensor(1).uniform_(0.4, 0.6))

            inp_network_noise = torch.randn(batch_size, self.noise_input_channel, self.unet_dim, self.unet_dim).to(self.device)

            # paper says to add noise augmentation to input noise during inference
            inp_network_noise = smoothen_image(inp_network_noise, sigma).to(self.device)

            # concatenating noise with rgb agnostic image across channels
            # corrupt -> concatenate -> predict
            x = torch.cat((inp_network_noise, ia), dim=1).to(self.device)

            for i in reversed(range(1, self.time_steps)):
                t = (torch.ones(batch_size) * i).long().to(self.device)
                predicted_noise = model(x, ic, jp, jg, t, sigma).to(self.device)

                # ToDo: Add Classifier-Free Guidance with guidance weight 2
                alpha = self.alpha[t][:, None, None, None].to(self.device)
                alpha_cumprod = self.alpha_cumprod[t][:, None, None, None].to(self.device)
                beta = self.beta[t][:, None, None, None].to(self.device)

                if i > 1:
                    noise = torch.randn_like(inp_network_noise).to(self.device)
                else:
                    noise = torch.zeros_like(inp_network_noise).to(self.device)

                inp_network_noise = 1 / torch.sqrt(alpha) * (inp_network_noise - ((1 - alpha) / (torch.sqrt(1 - alpha_cumprod))) * predicted_noise) + torch.sqrt(beta) * noise
        inp_network_noise = (inp_network_noise.clamp(-1, 1) + 1) / 2
        inp_network_noise = (inp_network_noise * 255).type(torch.uint8)

        return inp_network_noise

    def prepare(self, args):
        # 准备训练所需要的数据加载器、优化器、EMA指数移动平均

        train_dataset = UNetDataset(ip_dir=args.train_ip_folder,
                                    jp_dir=args.train_jp_folder,
                                    jg_dir=args.train_jg_folder,
                                    ia_dir=args.train_ia_folder,
                                    ic_dir=args.train_ic_folder,
                                    itr128_dir=args.train_itr128_folder,
                                    unet_size=self.unet_dim)

        validation_dataset = UNetDataset(ip_dir=args.validation_ip_folder,
                                         jp_dir=args.validation_jp_folder,
                                         jg_dir=args.validation_jg_folder,
                                         ia_dir=args.validation_ia_folder,
                                         ic_dir=args.validation_ic_folder,
                                         itr128_dir=args.validation_itr128_folder,
                                         unet_size=self.unet_dim)

        self.train_dataloader = DataLoader(train_dataset, args.batch_size_train, shuffle=False)
        self.val_dataloader = DataLoader(validation_dataset, args.batch_size_validation, shuffle=False)
        self.optimizer = optim.AdamW(self.net.parameters(), lr=args.lr, eps=1e-4)
        self.scheduler = schedule_lr(total_steps=round((args.data_len * args.epochs) / args.batch_size_train),
                                     start_lr=args.start_lr, stop_lr=args.stop_lr,
                                     pct_increasing_lr=args.pct_increasing_lr)
        self.mse = nn.MSELoss()
        self.ema = EMA(self.beta_ema)
        self.scaler = torch.cuda.amp.GradScaler()

    def prepare_for_inference(self, args):
        # 单卡 GPU 模式加载模型
        # self.net.load_state_dict(torch.load(args.model_path, map_location=args.device))
        # 多卡 DP 模式加载模型 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        self.net.load_state_dict({k.replace('module.',''):v for k,v in torch.load(args.model_path).items()})
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # 设置模型为评估模式
        self.net.eval()

        # 如果您有使用自编码器（如 FC1 和 FC2）也需要加载它们的状态
        self.fc1.load_state_dict(torch.load(args.fc1_model_path, map_location=args.device))
        self.fc2.load_state_dict(torch.load(args.fc2_model_path, map_location=args.device))

        
    # 进行单个训练步骤，包括反向传播和参数更新
    def train_step(self, loss, running_step):
        # 执行单个训练步骤，包括反向传播和参数更新

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.ema.step_ema(self.ema_net, self.net)

        for g in self.optimizer.param_groups:
            g['lr'] = self.scheduler[running_step]

    def single_epoch(self, unet_dim=128, epoch=-1, epochs=-1, every_epoch_steps=-1, train=True):
        # 处理一个训练/验证周期，并可以选择性的打印出损失

        total_loss = 0.
        avg_loss = 0.
        num_batches = 0

        if train:
            self.net.train()
            dataloader = self.train_dataloader  # 使用训练数据加载器
        else:
            self.net.eval()
            dataloader = self.val_dataloader  # 使用验证数据加载器

        for ip, jp, jg, ia, ic, itr128 in dataloader:
            # 这里是针对每个 epoch 过大，在中间 ?% 进行一步模型权重存储临时使用
            # if train == True:
            #     if self.running_train_steps == round(0.2 * every_epoch_steps):
            #         print("Now temp save checkpoints.")
            #         self.save_models(0, self.unet_dim)

            # noise augmentation
            # 在任何其他处理之前，向 ia、ic 添加随机高斯噪声进行噪声增强
            sigma = float(torch.FloatTensor(1).uniform_(0.4, 0.6))

            # 对于图像数据，使用列表推导式处理批次中的每个样本
            ia_batch = torch.cat([smoothen_image(create_transforms_imgs(read_img(path), unet_dim).unsqueeze(0).to(self.device), sigma) for path in ia])
            ic_batch = torch.cat([smoothen_image(create_transforms_imgs(read_img(path), unet_dim).unsqueeze(0).to(self.device), sigma) for path in ic])
            ip_batch = torch.cat([create_transforms_imgs(read_img(path), unet_dim).unsqueeze(0).to(self.device) for path in ip])

            if (unet_dim == 256):
                itr128_batch = torch.cat([smoothen_image(create_transforms_imgs(read_img(path), unet_dim).unsqueeze(0).to(self.device), sigma) for path in itr128])

            # 这里得到的是一个 jp json 的文件路径，先读取 json 的内容，转换成 tensor，再通过 FC 网络处理
            jp_data = []
            for jp_item in jp:
                with open(jp_item, 'r') as jp_item:
                    jp_json = json.load(jp_item)
                    jp_json_normalize = normalize_lst(jp_json)
                    jp_data.append(jp_json_normalize)
            jp_tensor = torch.tensor(jp_data)
            jp_fc1 = self.fc1(jp_tensor)
            jp = jp_fc1[1].clone().detach().to(self.device)
            
            jg_data = []
            for jg_item in jg:
                with open(jg_item, 'r') as jg_item:
                    jg_json = json.load(jg_item)
                    jg_json_normalize = normalize_lst(jg_json)
                    jg_data.append(jg_json_normalize)
            jg_tensor = torch.tensor(jg_data)
            jg_fc2 = self.fc2(jg_tensor)
            jg = jg_fc2[1].clone().detach().to(self.device)

            with torch.autocast(self.device) and (torch.inference_mode() if not train else torch.enable_grad()):
                t = self.sample_time_steps(ip_batch.shape[0]).to(self.device)

                # corrupt -> concatenate -> predict
                # 对 ip 添加 noise 变成 zt
                zt, noise_epsilon = self.add_noise_to_img(ip_batch, t)

                # unet128: ia 与 zt 进行 concat，用 zt 表示，准备将数据输入网络中
                if (unet_dim == 128):
                    zt = torch.cat((ia_batch, zt), dim=1).to(self.device)
                # unet256: itr128, ia 与 zt 进行 concat，用 zt 表示，准备将数据输入网络中
                elif (unet_dim == 256):
                    zt = torch.cat((ia_batch, zt, itr128_batch), dim=1).to(self.device)

                # 执行具体的网络
                predicted_noise = self.net(zt, ic_batch, jp, jg, t, sigma)
                loss = self.mse(noise_epsilon, predicted_noise)
                total_loss += loss.item()
                num_batches += 1  # 记录批次数量

            if train:
                print("Epoch: " + str(epoch+1) + "/" + str(epochs) + " ::: " + "Step: " + str(self.running_train_steps) + "/" + str(every_epoch_steps)+" ==================== "+f"train_mse_loss: {loss.item():2.3f}, learning_rate: {self.scheduler[self.running_train_steps]}")
                self.train_step(loss, self.running_train_steps)
                self.running_train_steps += 1
        
        avg_loss = total_loss / num_batches  # 计算平均损失
        return avg_loss

    def logging_images(self, unet_dim, epoch=-1, train=False):
        # 在训练时记录图像样本
        if (train == True):
            for idx, (ip, jp, jg, ia, ic, itr128) in enumerate(self.train_dataloader): # 这里一次拿到一个批次
                for i in range(len(ip)): # 这里需要从批次里把每张图片拿出来
                    # 获取 ip 的文件名
                    person_name = os.path.basename(ip[i])[:-4]

                    ia_item = read_img(ia[i])
                    ia_item = create_transforms_imgs(ia_item, unet_dim)
                    ia_item = ia_item.clone().detach()
                    ia_item.unsqueeze_(0)

                    ic_item = read_img(ic[i])
                    ic_item = create_transforms_imgs(ic_item, unet_dim)
                    ic_item = ic_item.clone().detach()
                    ic_item.unsqueeze_(0)

                    ip_item = read_img(ip[i])
                    ip_item = create_transforms_imgs(ip_item, unet_dim)
                    ip_item = ip_item.clone().detach()
                    ip_item.unsqueeze_(0)

                    if(unet_dim == 256):
                        itr128_item = read_img(itr128[i])
                        itr128_item = create_transforms_imgs(itr128_item, unet_dim)
                        itr128_item = itr128_item.clone().detach()
                        itr128_item.unsqueeze_(0)

                    # 这里得到的是一个 jp json 的文件路径，先读取 json 的内容，转换成 tensor，再通过 FC 网络处理
                    jp_data = []
                    with open(jp[i], 'r') as jp_item:
                        jp_json = json.load(jp_item)
                        jp_json_normalize = normalize_lst(jp_json)
                        jp_data.append(jp_json_normalize)
                    jp_tensor = torch.tensor(jp_data)
                    jp_fc1 = self.fc1(jp_tensor)
                    jp_item = jp_fc1[1].clone().detach().to(self.device)
                    
                    jg_data = []
                    with open(jg[i], 'r') as jg_item:
                        jg_json = json.load(jg_item)
                        jg_json_normalize = normalize_lst(jg_json)
                        jg_data.append(jg_json_normalize)
                    jg_tensor = torch.tensor(jg_data)
                    jg_fc2 = self.fc2(jg_tensor)
                    jg_item = jg_fc2[1].clone().detach().to(self.device)

                    # sampled image
                    sampled_image = self.sample(use_ema=False, conditional_inputs=(ia_item, ic_item, jp_item, jg_item))
                    sampled_image = sampled_image[0].permute(1, 2, 0).squeeze().cpu().numpy()

                    # ema sampled image
                    ema_sampled_image = self.sample(use_ema=True, conditional_inputs=(ia_item, ic_item, jp_item, jg_item))
                    ema_sampled_image = ema_sampled_image[0].permute(1, 2, 0).squeeze().cpu().numpy()

                    # 保存 itr128 或者 itr256
                    itr_folder = os.path.join("data/train", f"itr{unet_dim}")
                    itrema_folder = os.path.join("data/train", f"itr{unet_dim}_ema")
                    # save sampled image
                    cv2.imwrite(os.path.join(itr_folder, f"{person_name}.jpg"), sampled_image)
                    # save ema sampled image
                    cv2.imwrite(os.path.join(itrema_folder, f"{person_name}.jpg"), ema_sampled_image)
                    print(f"In train: Saved itr_{unet_dim} {person_name}.jpg")

        # 在验证时记录图像样本
        elif (train == False):
            for idx, (ip, jp, jg, ia, ic, itr128) in enumerate(self.val_dataloader):  # 这里一次拿到一个批次
                for i in range(len(ip)):  # 这里需要从批次里把每张图片拿出来
                    # 获取 ip 的文件名
                    person_name = os.path.basename(ip[i])[:-4]

                    ia_item = read_img(ia[i])
                    ia_item = create_transforms_imgs(ia_item, unet_dim)
                    ia_item = ia_item.clone().detach()
                    ia_item.unsqueeze_(0)

                    ic_item = read_img(ic[i])
                    ic_item = create_transforms_imgs(ic_item, unet_dim)
                    ic_item = ic_item.clone().detach()
                    ic_item.unsqueeze_(0)

                    ip_item = read_img(ip[i])
                    ip_item = create_transforms_imgs(ip_item, unet_dim)
                    ip_item = ip_item.clone().detach()
                    ip_item.unsqueeze_(0)

                    if(unet_dim == 256):
                        itr128_item = read_img(itr128[i])
                        itr128_item = create_transforms_imgs(itr128_item, unet_dim)
                        itr128_item = itr128_item.clone().detach()
                        itr128_item.unsqueeze_(0)

                    # 这里得到的是一个 jp json 的文件路径，先读取 json 的内容，转换成 tensor，再通过 FC 网络处理
                    jp_data = []
                    with open(jp[i], 'r') as jp_item:
                        jp_json = json.load(jp_item)
                        jp_json_normalize = normalize_lst(jp_json)
                        jp_data.append(jp_json_normalize)
                    jp_tensor = torch.tensor(jp_data)
                    jp_fc1 = self.fc1(jp_tensor)
                    jp_item = jp_fc1[1].clone().detach().to(self.device)
                    
                    jg_data = []
                    with open(jg[i], 'r') as jg_item:
                        jg_json = json.load(jg_item)
                        jg_json_normalize = normalize_lst(jg_json)
                        jg_data.append(jg_json_normalize)
                    jg_tensor = torch.tensor(jg_data)
                    jg_fc2 = self.fc2(jg_tensor)
                    jg_item = jg_fc2[1].clone().detach().to(self.device)

                    # sampled image
                    sampled_image = self.sample(use_ema=False, conditional_inputs=(ia_item, ic_item, jp_item, jg_item))
                    sampled_image = sampled_image[0].permute(1, 2, 0).squeeze().cpu().numpy()

                    # ema sampled image
                    ema_sampled_image = self.sample(use_ema=True, conditional_inputs=(ia_item, ic_item, jp_item, jg_item))
                    ema_sampled_image = ema_sampled_image[0].permute(1, 2, 0).squeeze().cpu().numpy()

                    # base images
                    ip_np = ip_item[0].permute(1, 2, 0).squeeze().cpu().numpy()
                    ic_np = ic_item[0].permute(1, 2, 0).squeeze().cpu().numpy()
                    ia_np = ia_item[0].permute(1, 2, 0).squeeze().cpu().numpy()

                    # 保存 itr128 或者 itr256
                    itr_folder = os.path.join("data/val", f"itr{unet_dim}")
                    itrema_folder = os.path.join("data/val", f"itr{unet_dim}_ema")
                    # save sampled image
                    cv2.imwrite(os.path.join(itr_folder, f"{person_name}.jpg"), sampled_image)
                    # save ema sampled image
                    cv2.imwrite(os.path.join(itrema_folder, f"{person_name}.jpg"), ema_sampled_image)
                    print(f"In val: Saved itr_{unet_dim} {person_name}.jpg")

                    # 保存验证的图像
                    os.makedirs(os.path.join("runs", f"{epoch+1}"), exist_ok=True)
                    # define folder paths
                    images_folder = os.path.join("runs", f"{epoch+1}", f"{person_name}")
                    # save base images
                    cv2.imwrite(os.path.join(images_folder, "person.jpg"), ip_np)
                    cv2.imwrite(os.path.join(images_folder, "seg_garment.jpg"), ic_np)
                    cv2.imwrite(os.path.join(images_folder, "seg_person.jpg"), ia_np)
                    # save sampled image
                    cv2.imwrite(os.path.join(images_folder, "sampled.jpg"), sampled_image)
                    # save ema sampled image
                    cv2.imwrite(os.path.join(images_folder, "ema_sampled.jpg"), ema_sampled_image)
                    print(f"In val: Saved epoch:{epoch+1} images")

    def save_models(self, epoch=-1, unet_dim=128):
        # 保存模型的权重和优化器状态

        print(f"Save models epoch: {epoch+1}.")

        # 模型保存目录
        ckpt_path = os.path.join("tmp_models", f"ckpt{unet_dim}")
        ema_ckpt_path = os.path.join("tmp_models", f"ema_ckpt{unet_dim}")
        optim_path = os.path.join("tmp_models", f"optim{unet_dim}")
        # 若目录不存在就创建
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)
        if not os.path.exists(ema_ckpt_path):
            os.makedirs(ema_ckpt_path)
        if not os.path.exists(optim_path):
            os.makedirs(optim_path)
        
        # 单卡GPU训练
        torch.save(self.net.state_dict(), os.path.join(ckpt_path, f"ckpt_{epoch+1}.pt"))
        torch.save(self.ema_net.state_dict(), os.path.join(ema_ckpt_path, f"ema_ckpt_{epoch+1}.pt"))
        torch.save(self.optimizer.state_dict(), os.path.join(optim_path, f"optim_{epoch+1}.pt"))

    def fit(self, args):
        # 开始和管理训练过程

        print(f"Starting training")
        print("args.total_steps: ", round((args.data_len * args.epochs) / args.batch_size_train))
        print("args.batch_size_train: ", args.batch_size_train)
        print("data_len: ", args.data_len)
        print("\nEpochs: ", args.epochs)

        if args.epochs < 0:
            args.epochs = 1

        # 每一个 epoch 包含多少 steps
        every_epoch_steps = round(args.data_len / args.batch_size_train)
        print("Every epoch steps: ", every_epoch_steps)

        for epoch in range(args.epochs):
            print(f"\nStarting Epoch: {epoch + 1} / {args.epochs}")
            self.running_train_steps = 1
            # 记录这一个 epoch 开始时间
            start_time = time.time()
            # 具体执行新的一轮
            # 每个 epoch 都会对 itr 进行保存：itr128, itr256
            train_loss = self.single_epoch(unet_dim=self.unet_dim, epoch=epoch, epochs=args.epochs, every_epoch_steps=every_epoch_steps, train=True)
            print(f"Epoch {epoch+1}: Training Loss: {train_loss}")
            # 记录这一个 epoch 结束时间
            end_time = time.time()
            # 计算这一个 epoch 总花费时间
            total_time = end_time - start_time
            # Convert the time into hours, minutes, and seconds
            hours = int(total_time // 3600)
            minutes = int((total_time % 3600) // 60)
            seconds = total_time % 60
            print(f"Epoch Execution Time: {hours} hours, {minutes} minutes, and {seconds:.2f} seconds")

            # 函数中指定多少个 epoch 执行完后，对结果进行保存：seg_person.png, seg_garment.png, person.png, sample.png, ema_sample.png
            # 每个周期进行验证，并记录验证集的图像

            # 在每个训练周期结束后进行验证
            val_loss = self.single_epoch(unet_dim=self.unet_dim, epoch=epoch, epochs=args.epochs, every_epoch_steps=every_epoch_steps, train=False)
            print(f"Epoch {epoch+1}: Validation Loss: {val_loss}")

            # 指定多少个 epoch 执行完后，对模型进行保存：ckpt.pth, ema_ckpt.pth, optim.pth
            if (epoch + 1) % args.model_saving_frequency == 0:
                self.save_models(epoch, self.unet_dim)
            
            if (epoch + 1) == 1 or (epoch + 1) % 50 == 0:
                # 训练结束，对训练集进行图像记录
                self.logging_images(unet_dim=self.unet_dim, epoch=epoch, train=True)

            if (epoch + 1) == 1 or (epoch + 1) % 50 == 0:
                # 训练结束，对验证集进行图像记录
                self.logging_images(unet_dim=self.unet_dim, epoch=epoch, train=False)

        print(f"Training Done Successfully!")
