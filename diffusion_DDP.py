import copy
import json
import os
import cv2
import time
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch import optim
import torch.distributed as dist
from torch.cuda.amp import GradScaler
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.utils import GaussianSmoothing, read_img
from utils.dataloader_train import UNetDataset, create_transforms_imgs
from pre_processing.garment_pose_embedding.utils.dataloader import normalize_lst
from pre_processing.person_pose_embedding.network import AutoEncoder as PersonAutoEncoder
from pre_processing.garment_pose_embedding.network import AutoEncoder as GarmentAutoEncoder
#from ema import EMA
from UNet128 import UNet128
from UNet256 import UNet256


def smoothen_image(img, sigma, device):
    # As suggested in: https://jmlr.csail.mit.edu/papers/volume23/21-0635/21-0635.pdf Section 4.4
    # 高斯噪音增强函数
    # 输入：
    # • img：输入图像
    # • sigma：高斯核的标准差

    smoothing2d = GaussianSmoothing(channels=3,
                                    kernel_size=3,
                                    sigma=sigma,
                                    conv_dim=2)
    smoothing2d = smoothing2d.to(device)

    img = F.pad(img, (1, 1, 1, 1), mode='reflect')
    with torch.no_grad():
        img = smoothing2d(img)

    return img

def calculate_time(start_time, end_time):
    # 计算时间的函数
    total_time = end_time - start_time
    # Convert the time into hours, minutes, and seconds
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)

    return hours, minutes, seconds

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

    def __init__(self, args, gpuid, unet_dim=128, pose_embed_dim=8, time_steps=256, beta_start=1e-4, beta_end=0.02, beta_ema=0.995, noise_input_channel=3):
        self.device = f'cuda:{gpuid}'
        self.unet_dim = unet_dim
        self.time_steps = time_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        #self.beta_ema = beta_ema
        self.noise_input_channel = noise_input_channel
        self.beta = self.linear_beta_scheduler().to(self.device)
        self.alpha = 1 - self.beta
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0)
        dist.init_process_group(backend='nccl', init_method='env://')
        args.rank = dist.get_rank()
        # 在任何其他处理之前，向 ia、ic 添加随机高斯噪声进行噪声增强
        self.sigma = float(torch.FloatTensor(1).uniform_(0.4, 0.6))
        
        # 确定网络模型
        if unet_dim == 128:
            self.net = UNet128(pose_embed_dim, time_steps, self.sigma)
        elif unet_dim == 256:
            self.net = UNet256(pose_embed_dim, time_steps, self.sigma)
        
        # 将网络模型移到设备上
        self.net = self.net.to(self.device)

        self.net = nn.SyncBatchNorm.convert_sync_batchnorm(self.net)
        self.net = nn.parallel.DistributedDataParallel(self.net, device_ids=[gpuid])
        
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

        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        self.train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size_train, shuffle=False, num_workers=8, pin_memory=False, sampler=train_sampler)

        val_sampler = torch.utils.data.distributed.DistributedSampler(validation_dataset)
        self.val_dataloader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=args.batch_size_validation, shuffle=False, num_workers=8, pin_memory=False, sampler=val_sampler)

        self.data_len = len(self.train_dataloader.sampler) * args.batch_size_train
        self.optimizer = optim.AdamW(self.net.parameters(), lr=args.lr, eps=1e-4)
        self.scheduler = schedule_lr(total_steps=round((self.data_len * args.epochs) / args.batch_size_train),
                                     start_lr=args.start_lr, stop_lr=args.stop_lr,
                                     pct_increasing_lr=args.pct_increasing_lr)
        self.mse = nn.MSELoss()
        #self.ema = EMA(self.beta_ema)
        self.scaler = GradScaler(enabled=args.use_mix_precision)

        self.fc1 = PersonAutoEncoder(34)
        self.fc1.load_state_dict(torch.load(args.fc1_model_path, map_location=self.device))
        self.fc1.eval()
        self.fc2 = GarmentAutoEncoder(34)
        self.fc2.load_state_dict(torch.load(args.fc2_model_path, map_location=self.device))
        self.fc2.eval()

        #self.ema_net = copy.deepcopy(self.net).eval().requires_grad_(False)

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
        return (sqrt_alpha_timestep * img) + (sqrt_one_minus_alpha_timestep * epsilon), epsilon

    @torch.inference_mode()
    def sample(self, use_ema, conditional_inputs):
        # 生成样本图像
        # 参数：
        # • use_ema: 布尔值，指示是否使用EMA（指数移动平均）模型来生成图像。
        # • conditional_inputs: 条件输入，包括人物姿势、服装姿势和其他相关信息

        #model = self.ema_net if use_ema else self.net
        model = self.net
        ia, ic, jp, jg = conditional_inputs

        ia = ia.to(self.device)
        ic = ic.to(self.device)
        jp = jp.to(self.device)
        jg = jg.to(self.device)
        batch_size = len(ic)
        # print(f"Running inference for {batch_size} images")

        model.eval()
        with torch.inference_mode():
            inp_network_noise = torch.randn(batch_size, self.noise_input_channel, self.unet_dim, self.unet_dim).to(self.device)

            # paper says to add noise augmentation to input noise during inference
            inp_network_noise = smoothen_image(inp_network_noise, self.sigma)

            # concatenating noise with rgb agnostic image across channels
            # corrupt -> concatenate -> predict
            x = torch.cat((ia,inp_network_noise), dim=1).to(self.device)

            for i in reversed(range(1, self.time_steps)):
                t = (torch.ones(batch_size) * i).long().to(self.device)
                predicted_noise = model(x, ic, jp, jg, t).to(self.device)

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
        
    # 进行单个训练步骤，包括反向传播和参数更新
    def train_step(self, loss, running_step):
        # 执行单个训练步骤，包括反向传播和参数更新

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        #self.ema.step_ema(self.ema_net, self.net)

        for g in self.optimizer.param_groups:
            g['lr'] = self.scheduler[running_step]

    def single_epoch(self, args, unet_dim=128, epoch=-1, epochs=-1, every_epoch_steps=-1):
        # 处理一个训练/验证周期，并可以选择性的打印出损失

        total_loss = 0.
        train_loss = 0.
        num_batches = 0

        self.net.train()
        dataloader = self.train_dataloader  # 使用训练数据加载器
        dataloader.sampler.set_epoch(epoch)

        for ip, jp, jg, ia, ic, itr128 in (tqdm(dataloader) if args.rank == 0 else dataloader):
            torch.cuda.empty_cache()

            # 这里是针对每个 epoch 过大，在中间 ?% 进行一步模型权重存储临时使用
            # if self.running_train_steps == round(0.01 * every_epoch_steps):
            #     if dist.get_rank() == 0:
            #         print("Now temp save checkpoints.")
            #     self.save_models(args.save_model_path, 0, self.unet_dim)

            # 对于图像数据，使用列表推导式处理批次中的每个样本
            with torch.no_grad():
                ia_batch = torch.cat([smoothen_image(create_transforms_imgs(read_img(path), unet_dim).unsqueeze(0).to(self.device), self.sigma, self.device) for path in ia])
                ic_batch = torch.cat([smoothen_image(create_transforms_imgs(read_img(path), unet_dim).unsqueeze(0).to(self.device), self.sigma, self.device) for path in ic])
                ip_batch = torch.cat([create_transforms_imgs(read_img(path), unet_dim).unsqueeze(0).to(self.device) for path in ip])

                if (unet_dim == 256):
                    itr128_batch = torch.cat([smoothen_image(create_transforms_imgs(read_img(path), unet_dim).unsqueeze(0).to(self.device), self.sigma, self.device) for path in itr128])

            # 这里得到的是一个 jp json 的文件路径，先读取 json 的内容，转换成 tensor，再通过 FC 网络处理
            with torch.no_grad():
                jp_data = []
                for jp_item in jp:
                    with open(jp_item, 'r') as jp_item:
                        jp_json = json.load(jp_item)
                        jp_json_normalize = normalize_lst(jp_json)
                        jp_data.append(jp_json_normalize)
                jp_tensor = torch.tensor(jp_data)
                jp_fc1 = self.fc1(jp_tensor)
                jp = jp_fc1[1]
            
                jg_data = []
                for jg_item in jg:
                    with open(jg_item, 'r') as jg_item:
                        jg_json = json.load(jg_item)
                        jg_json_normalize = normalize_lst(jg_json)
                        jg_data.append(jg_json_normalize)
                jg_tensor = torch.tensor(jg_data)
                jg_fc2 = self.fc2(jg_tensor)
                jg = jg_fc2[1]

            with torch.autocast('cuda') and torch.enable_grad():
                t = self.sample_time_steps(ip_batch.shape[0])

                # corrupt -> concatenate -> predict
                # 对 ip 添加 noise 变成 zt
                zt, noise_epsilon = self.add_noise_to_img(ip_batch, t)

                # unet128: ia 与 zt 进行 concat，用 zt 表示，准备将数据输入网络中
                # person-UNet 将ia和噪声图像 𝐳t作为输入
                # 由于ia 和 𝐳t是按像素对齐的，因此我们在 UNet 处理开始时直接沿通道维度将它们连接起来
                if (unet_dim == 128):
                    zt = torch.cat((ia_batch, zt), dim=1)
                # unet256: itr128, ia 与 zt 进行 concat，用 zt 表示，准备将数据输入网络中
                elif (unet_dim == 256):
                    zt = torch.cat((ia_batch, zt, itr128_batch), dim=1)

                # 利用torch.cuda.amp.autocast控制前向过程中是否使用半精度计算
                # garment-UNet 将ic作为输入
                with torch.cuda.amp.autocast(enabled=args.use_mix_precision):
                    predicted_noise = self.net(zt, ic_batch, jp, jg, t)
                    loss = self.mse(noise_epsilon, predicted_noise)
                    total_loss += loss.item()
                num_batches += 1  # 记录批次数量

            self.train_step(loss, self.running_train_steps)
            self.running_train_steps += 1
        
        train_loss = total_loss / num_batches  # 计算平均损失

        # 只在 rank 0 上打印平均损失
        if dist.get_rank() == 0:
            print(f"Epoch {epoch+1}: Training Loss: {train_loss}")

        return train_loss
    
    def evaluate(self, unet_dim=128, epoch=-1):
        # 处理一个训练/验证周期，并可以选择性的打印出损失

        total_loss = 0.
        val_loss = 0.
        num_batches = 0

        self.net.eval()
        dataloader = self.val_dataloader  # 使用验证数据加载器

        for ip, jp, jg, ia, ic, itr128 in dataloader:

            # 对于图像数据，使用列表推导式处理批次中的每个样本
            with torch.no_grad():
                ia_batch = torch.cat([smoothen_image(create_transforms_imgs(read_img(path), unet_dim).unsqueeze(0).to(self.device), self.sigma, self.device) for path in ia])
                ic_batch = torch.cat([smoothen_image(create_transforms_imgs(read_img(path), unet_dim).unsqueeze(0).to(self.device), self.sigma, self.device) for path in ic])
                ip_batch = torch.cat([create_transforms_imgs(read_img(path), unet_dim).unsqueeze(0).to(self.device) for path in ip])

            if (unet_dim == 256):
                with torch.no_grad():
                    itr128_batch = torch.cat([smoothen_image(create_transforms_imgs(read_img(path), unet_dim).unsqueeze(0).to(self.device), self.sigma, self.device) for path in itr128])
            
            with torch.no_grad():
                # 这里得到的是一个 jp json 的文件路径，先读取 json 的内容，转换成 tensor，再通过 FC 网络处理
                jp_data = []
                for jp_item in jp:
                    with open(jp_item, 'r') as jp_item:
                        jp_json = json.load(jp_item)
                        jp_json_normalize = normalize_lst(jp_json)
                        jp_data.append(jp_json_normalize)
                jp_tensor = torch.tensor(jp_data)
                jp_fc1 = self.fc1(jp_tensor)
                jp = jp_fc1[1]
                
                jg_data = []
                for jg_item in jg:
                    with open(jg_item, 'r') as jg_item:
                        jg_json = json.load(jg_item)
                        jg_json_normalize = normalize_lst(jg_json)
                        jg_data.append(jg_json_normalize)
                jg_tensor = torch.tensor(jg_data)
                jg_fc2 = self.fc2(jg_tensor)
                jg = jg_fc2[1]

            with torch.autocast('cuda') and torch.inference_mode():
                t = self.sample_time_steps(ip_batch.shape[0])

                # corrupt -> concatenate -> predict
                # 对 ip 添加 noise 变成 zt
                zt, noise_epsilon = self.add_noise_to_img(ip_batch, t)

                # unet128: ia 与 zt 进行 concat，用 zt 表示，准备将数据输入网络中
                if (unet_dim == 128):
                    zt = torch.cat((ia_batch, zt), dim=1)
                # unet256: itr128, ia 与 zt 进行 concat，用 zt 表示，准备将数据输入网络中
                elif (unet_dim == 256):
                    zt = torch.cat((ia_batch, zt, itr128_batch), dim=1)

                # 执行具体的网络
                predicted_noise = self.net(zt, ic_batch, jp, jg, t)
                loss = self.mse(noise_epsilon, predicted_noise)
                total_loss += loss.item()
                num_batches += 1  # 记录批次数量
        
        # 将 total_loss 转换为张量，并在所有 GPU 上聚合
        total_loss_tensor = torch.tensor(total_loss).to(self.device)
        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)

        # 确保 num_batches 在所有 GPU 上相同
        num_batches_tensor = torch.tensor(num_batches).to(self.device)
        dist.all_reduce(num_batches_tensor, op=dist.ReduceOp.SUM)

        # 在每个 GPU 上计算平均损失
        val_loss = total_loss_tensor.item() / num_batches_tensor.item()

        # 只在 rank 0 上打印平均损失
        if dist.get_rank() == 0:
            print(f"Epoch {epoch+1}: Validation Loss: {val_loss}")

        return val_loss

    def logging_images(self, unet_dim, epoch=-1, train=False):
        # 记录图像样本

        if train:
            dataloader = self.train_dataloader
        else:
            dataloader = self.val_dataloader

        for idx, (ip, jp, jg, ia, ic, itr128) in enumerate(dataloader): # 这里一次拿到一个批次
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
                with torch.no_grad():
                    jp_data = []
                    with open(jp[i], 'r') as jp_item:
                        jp_json = json.load(jp_item)
                        jp_json_normalize = normalize_lst(jp_json)
                        jp_data.append(jp_json_normalize)
                    jp_tensor = torch.tensor(jp_data)
                    jp_fc1 = self.fc1(jp_tensor)
                    jp_item = jp_fc1[1]
                    
                    jg_data = []
                    with open(jg[i], 'r') as jg_item:
                        jg_json = json.load(jg_item)
                        jg_json_normalize = normalize_lst(jg_json)
                        jg_data.append(jg_json_normalize)
                    jg_tensor = torch.tensor(jg_data)
                    jg_fc2 = self.fc2(jg_tensor)
                    jg_item = jg_fc2[1]

                # sampled image
                sampled_image = self.sample(use_ema=False, conditional_inputs=(ia_item, ic_item, jp_item, jg_item))
                sampled_image = sampled_image[0].permute(1, 2, 0).squeeze().cpu().numpy()

                # ema sampled image
                # ema_sampled_image = self.sample(use_ema=True, conditional_inputs=(ia_item, ic_item, jp_item, jg_item))
                # ema_sampled_image = ema_sampled_image[0].permute(1, 2, 0).squeeze().cpu().numpy()

                if train:
                    # 保存 itr128 或者 itr256
                    itr_folder = os.path.join("data/train", f"itr{unet_dim}")
                    # itrema_folder = os.path.join("data/train", f"itr{unet_dim}_ema")
                    # save sampled image
                    cv2.imwrite(os.path.join(itr_folder, f"{person_name}.jpg"), sampled_image)
                    # save ema sampled image
                    # cv2.imwrite(os.path.join(itrema_folder, f"{person_name}.jpg"), ema_sampled_image)
                    print(f"In train: Saved itr_{unet_dim} {person_name}.jpg")
                else:
                    # base images
                    ip_np = ip_item[0].permute(1, 2, 0).squeeze().cpu().numpy()
                    ic_np = ic_item[0].permute(1, 2, 0).squeeze().cpu().numpy()
                    ia_np = ia_item[0].permute(1, 2, 0).squeeze().cpu().numpy()

                    # 保存 itr128 或者 itr256
                    itr_folder = os.path.join("data/val", f"itr{unet_dim}")
                    # itrema_folder = os.path.join("data/val", f"itr{unet_dim}_ema")
                    # save sampled image
                    cv2.imwrite(os.path.join(itr_folder, f"{person_name}.jpg"), sampled_image)
                    # save ema sampled image
                    # cv2.imwrite(os.path.join(itrema_folder, f"{person_name}.jpg"), ema_sampled_image)
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
                    # cv2.imwrite(os.path.join(images_folder, "ema_sampled.jpg"), ema_sampled_image)
                    print(f"In val: Saved epoch:{epoch+1} images")

    def save_models(self, save_model_path, epoch=-1, unet_dim=128):
        # 保存模型的权重和优化器状态

        if dist.get_rank() == 0:
            print(f"Save models epoch: {epoch+1}.")

            # 模型保存目录
            ckpt_path = os.path.join(save_model_path, f"ckpt{unet_dim}")
            # ema_ckpt_path = os.path.join("tmp_models", f"ema_ckpt{unet_dim}")
            optim_path = os.path.join(save_model_path, f"optim{unet_dim}")
            # 若目录不存在就创建
            if not os.path.exists(ckpt_path):
                os.makedirs(ckpt_path)
            # if not os.path.exists(ema_ckpt_path):
            #     os.makedirs(ema_ckpt_path)
            if not os.path.exists(optim_path):
                os.makedirs(optim_path)
            
            torch.save(self.net.module.state_dict(), os.path.join(ckpt_path, f"ckpt_{epoch+1}.pth"))
            # torch.save(self.ema_net.state_dict(), os.path.join(ema_ckpt_path, f"ema_ckpt_{epoch+1}.pt"))
            torch.save(self.optimizer.state_dict(), os.path.join(optim_path, f"optim_{epoch+1}.pth"))

    def train(self, args):
        # 开始和管理训练过程

        # 每一个 epoch 包含多少 steps
        every_epoch_steps = round(self.data_len / args.batch_size_train)
        if dist.get_rank() == 0:
            print(f"\nStarting training\n")
            print("args.total_steps: ", round((self.data_len * args.epochs) / args.batch_size_train))
            print("args.batch_size_train: ", args.batch_size_train)
            print("data_len: ", self.data_len)
            print("Every epoch steps: ", every_epoch_steps)
            print("Epochs: ", args.epochs)
            writer = SummaryWriter(log_dir=args.logdir, flush_secs=120)

        for epoch in range(args.epochs):
            if dist.get_rank() == 0:
                print(f"\nStarting Epoch: {epoch + 1} / {args.epochs}")
            self.running_train_steps = 1
            # 记录这一个 epoch 开始时间
            start_time = time.time()
            # 在每个训练周期结束后进行验证，打印训练集 loss
            train_loss = self.single_epoch(args, unet_dim=self.unet_dim, epoch=epoch, epochs=args.epochs, every_epoch_steps=every_epoch_steps)
            if dist.get_rank() == 0:
                writer.add_scalar(tag='Loss/train', scalar_value=train_loss, global_step=epoch)
            # 记录这一个 epoch 结束时间
            end_time = time.time()
            # 计算这一个 epoch 总花费时间
            hours, minutes, seconds = calculate_time(start_time, end_time)
            if dist.get_rank() == 0:
                print(f"Epoch Execution Time: {hours} hours, {minutes} minutes, and {seconds} seconds")

            # 在每个训练周期结束后进行验证，打印验证集 loss
            val_loss = self.evaluate(unet_dim=self.unet_dim, epoch=epoch)
            if dist.get_rank() == 0:
                writer.add_scalar(tag='Loss/val', scalar_value=val_loss, global_step=epoch)

            # 指定多少个 epoch 执行完后，对模型进行保存：ckpt.pth, ema_ckpt.pth, optim.pth
            if (epoch + 1) % args.model_saving_frequency == 0:
                self.save_models(args.save_model_path, epoch, self.unet_dim)
            
            # 指定多少个 epoch 执行完后，对生成的训练集图像进行保存
            # if (epoch + 1) % 100 == 0:
            #     # 训练结束，对训练集进行图像记录
            #     # 记录开始时间
            #     start_time = time.time()
            #     self.logging_images(unet_dim=self.unet_dim, epoch=epoch, train=True)
            #     # 记录结束时间
            #     end_time = time.time()
            #     hours, minutes, seconds = calculate_time(start_time, end_time)
            #     if dist.get_rank() == 0:
            #         print(f"Train Save Time: {hours} hours, {minutes} minutes, and {seconds} seconds")

            # 指定多少个 epoch 执行完后，对生成的验证集图像进行保存
            # if (epoch + 1) % 100 == 0:
            #     # 训练结束，对验证集进行图像记录
            #     # 记录开始时间
            #     start_time = time.time()
            #     self.logging_images(unet_dim=self.unet_dim, epoch=epoch, train=False)
            #     # 记录结束时间
            #     end_time = time.time()
            #     hours, minutes, seconds = calculate_time(start_time, end_time)
            #     if dist.get_rank() == 0:
            #         print(f"Val Save Time: {hours} hours, {minutes} minutes, and {seconds} seconds")

        if dist.get_rank() == 0:
            writer.close()
            print(f"Training Done Successfully!")
        dist.destroy_process_group()
