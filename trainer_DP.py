from diffusion_DP import Diffusion


class ArgParser:
    def __init__(self):
        # 学习率，优化器相关参数
        self.lr = 0.0
        self.start_lr = 0.0
        self.stop_lr = 0.0001
        self.pct_increasing_lr = 0.02

        # 训练集
        self.train_ip_folder = "data_test/train/ip"  # 处理后变成zt
        self.train_jp_folder = "data_test/train/jp"
        self.train_ia_folder = "data_test/train/ia"
        self.train_ic_folder = "data_test/train/ic"
        self.train_jg_folder = "data_test/train/jg"
        self.train_itr128_folder = "data_test/train/itr128"

        # 验证集
        self.validation_ip_folder = "data_test/val/ip"  # 处理后变成zt
        self.validation_jp_folder = "data_test/val/jp"
        self.validation_ia_folder = "data_test/val/ia"
        self.validation_ic_folder = "data_test/val/ic"
        self.validation_jg_folder = "data_test/val/jg"
        self.validation_itr128_folder = "data_test/val/itr128"

        # 批次大小
        self.batch_size_train = 1  # 训练的 batch_size
        self.batch_size_validation = 1  # 验证的 batch_size

        # 保存频率，几个 epoch 保存一次
        self.calculate_loss_frequency = 1
        self.image_logging_frequency = 1
        self.model_saving_frequency = 10

        # 训练轮数
        self.epochs = 20

        # fc 模型
        self.fc1_model_path = '/home/xkmb/tryondiffusion/models/fc1.pth'  # FC1模型的路径
        self.fc2_model_path = '/home/xkmb/tryondiffusion/models/fc2.pth'  # FC2模型的路径


if __name__ == "__main__":
    args = ArgParser()
    diffusion = Diffusion(device="cuda", unet_dim=128, pose_embed_dim=8, time_steps=256, beta_start=1e-4, beta_end=0.02, beta_ema=0.995, noise_input_channel=3)
    diffusion.prepare(args)
    diffusion.train(args)
