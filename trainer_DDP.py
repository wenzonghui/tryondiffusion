import argparse
from diffusion_DDP import Diffusion

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpuid', default=0, type=int, help="which gpu to use")
    parser.add_argument('-e', '--epochs', default=10, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size_train', default=1, type=int, metavar='N', help='number of train batchsize')
    parser.add_argument('--batch_size_validation', default=1, type=int, metavar='N', help='number of val batchsize')
    parser.add_argument("--local_rank", type=int, help='rank in current node')
    parser.add_argument('--use_mix_precision', default=False, action='store_true', help="whether to use mix precision")
    parser.add_argument('-d', '--unet_dim', default=128, type=int, help="which net to train")
    
    # 学习率，优化器相关参数
    parser.add_argument('--lr', default=1e-4, type=float, help="")
    parser.add_argument('--start_lr', default=1e-4, type=float, help="")
    parser.add_argument('--stop_lr', default=1e-6, type=float, help="")
    parser.add_argument('--pct_increasing_lr', default=0.03, type=float, help="")
    
    # 训练集
    parser.add_argument('--train_ip_folder', default="data_test/train/ip", type=str, help="")
    parser.add_argument('--train_jp_folder', default="data_test/train/jp", type=str, help="")
    parser.add_argument('--train_ia_folder', default="data_test/train/ia", type=str, help="")
    parser.add_argument('--train_ic_folder', default="data_test/train/ic", type=str, help="")
    parser.add_argument('--train_jg_folder', default="data_test/train/jg", type=str, help="")
    parser.add_argument('--train_itr128_folder', default="data_test/train/itr128", type=str, help="")

    # 验证集
    parser.add_argument('--validation_ip_folder', default="data_test/val/ip", type=str, help="")
    parser.add_argument('--validation_jp_folder', default="data_test/val/jp", type=str, help="")
    parser.add_argument('--validation_ia_folder', default="data_test/val/ia", type=str, help="")
    parser.add_argument('--validation_ic_folder', default="data_test/val/ic", type=str, help="")
    parser.add_argument('--validation_jg_folder', default="data_test/val/jg", type=str, help="")
    parser.add_argument('--validation_itr128_folder', default="data_test/val/itr128", type=str, help="")

    # 保存频率，几个 epoch 保存一次
    parser.add_argument('--calculate_loss_frequency', default=1, type=int, help="")
    parser.add_argument('--image_logging_frequency', default=1, type=int, help="")
    parser.add_argument('--model_saving_frequency', default=10, type=int, help="")

    # FC 模型路径
    parser.add_argument('--fc1_model_path', default="/root/Desktop/dzy/models/fc1.pth", type=str, help="")
    parser.add_argument('--fc2_model_path', default="/root/Desktop/dzy/models/fc2.pth", type=str, help="")

    # log 记录
    parser.add_argument('--logdir', default="runs/unet128", type=str, help="")

    args = parser.parse_args()
    diffusion = Diffusion(args=args, gpuid=args.local_rank, unet_dim=args.unet_dim, pose_embed_dim=8, time_steps=256, beta_start=1e-4, beta_end=0.02, beta_ema=0.995, noise_input_channel=3)
    diffusion.prepare(args)
    diffusion.train(args)

    
if __name__ == "__main__":
    main()

    #################################################################################################################################################################
    # 一个节点（电脑）的情况下，启动方式
    # python -m torch.distributed.launch --nproc_per_node=8 trainer_DDP.py --use_mix_precision

    #################################################################################################################################################################

    #################################################################################################################################################################
    # 两台电脑，每台8卡情况下的启动方式
    # nproc_per_node: 每台机器中运行几个进程
    # nnodes：一共使用多少台机器
    # node_rank：当前机器的序号
    # master_addr：0号机器的IP
    # master_port：0号机器的可用端口

    # Node 0 : ip 192.168.1.201  port : 29501
    # terminal-0
    # python -m torch.distributed.launch --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr="192.168.1.202" --master_port=29501 trainer_DDP.py --use_mix_precision

    # Node 1 : 
    # terminal-0
    # python -m torch.distributed.launch --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr="192.168.1.202" --master_port=29501 trainer_DDP.py --use_mix_precision
    
    ##################################################################################################################################################################
