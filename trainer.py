from diffusion import Diffusion


class ArgParser:
    def __init__(self):
        self.unet_dim = 128
        # 学习率，优化器相关参数
        self.lr = 0
        self.start_lr = 0
        self.stop_lr = 1e-4
        self.pct_increasing_lr = 0.02
        self.use_mix_precision = False

        # 训练集
        self.train_ip_folder = "data/train/ip"  # 处理后变成zt
        self.train_jp_folder = "data/train/jp"
        self.train_ia_folder = "data/train/ia"
        self.train_ic_folder = "data/train/ic"
        self.train_jg_folder = "data/train/jg"
        self.train_itr128_folder = "data/train/itr128"

        # 验证集
        self.validation_ip_folder = "data/val/ip"  # 处理后变成zt
        self.validation_jp_folder = "data/val/jp"
        self.validation_ia_folder = "data/val/ia"
        self.validation_ic_folder = "data/val/ic"
        self.validation_jg_folder = "data/val/jg"
        self.validation_itr128_folder = "data/val/itr128"

        # 训练轮数
        self.epochs = 200

        # 批次大小
        self.batch_size_train = 6  # 训练的 batch_size
        self.batch_size_validation = 1  # 验证的 batch_size

        # 保存频率，几个 epoch 保存一次
        self.calculate_loss_frequency = 1
        self.image_logging_frequency = 1
        self.model_saving_frequency = 20

        # fc 模型路径
        self.fc1_model_path = '/root/Desktop/dzy/models/fc1.pth'  # FC1模型的路径
        self.fc2_model_path = '/root/Desktop/dzy/models/fc2.pth'  # FC2模型的路径

        # 模型保存路径
        self.save_model_path = "tmp_models"

        # log 记录
        self.logdir = "runs/unet128"




if __name__ == "__main__":
    args = ArgParser()
    diffusion = Diffusion(device="cuda", unet_dim=args.unet_dim, pose_embed_dim=8, time_steps=256, beta_start=1e-4, beta_end=0.02, beta_ema=0.995, noise_input_channel=3)
    diffusion.prepare(args)
    diffusion.train(args)
