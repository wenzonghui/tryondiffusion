from diffusion import Diffusion


class ArgParser:

    def __init__(self):
        self.run_name = "unet128"

        self.train_ip_folder = "data/train/ip"  # 处理后变成zt
        self.train_jp_folder = "data/train/jp"
        self.train_ia_folder = "data/train/ia"
        self.train_ic_folder = "data/train/ic"
        self.train_jg_folder = "data/train/jg"

        self.validation_ip_folder = "data/val/ip"  # 处理后变成zt
        self.validation_jp_folder = "data/val/jp"
        self.validation_ia_folder = "data/val/ia"
        self.validation_ic_folder = "data/val/ic"
        self.validation_jg_folder = "data/val/jg"

        self.batch_size_train = 8
        self.batch_size_validation = 1

        self.calculate_loss_frequency = 10
        self.image_logging_frequency = 10
        self.model_saving_frequency = 10

        self.total_steps = 100000
        self.lr = 0.0
        self.start_lr = 0.0
        self.stop_lr = 0.0001
        self.pct_increasing_lr = 0.02


if __name__ == "__main__":
    args = ArgParser()
    diffusion = Diffusion(device="cuda",
                          pose_embed_dim=8,
                          time_steps=256,
                          beta_start=1e-4,
                          beta_end=0.02,
                          unet_dim=128,
                          noise_input_channel=3,
                          beta_ema=0.995)

    diffusion.prepare(args)
    diffusion.fit(args)
