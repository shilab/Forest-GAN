

import argparse


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_epoch", type=int, default=200, help="number of epochs of training"
    )
    parser.add_argument(
        "--max_iter", type=int, default=20000, help="set the max iteration number"
    )
    parser.add_argument(
        "-gen_bs", "--gen_batch_size", type=int, default=64, help="size of the batches"
    )
    parser.add_argument(
        "-dis_bs", "--dis_batch_size", type=int, default=64, help="size of the batches"
    )
    parser.add_argument(
        "--g_lr", type=float, default=0.0002, help="adam: gen learning rate"
    )
    parser.add_argument(
        "--d_lr", type=float, default=0.0002, help="adam: disc learning rate"
    )
    parser.add_argument(
        "--ctrl_lr", type=float, default=3.5e-4, help="adam: ctrl learning rate"
    )
    parser.add_argument(
        "--lr_decay",  action="store_true", help="learning rate decay or not"
    )
    parser.add_argument(
        "--beta1",
        type=float,
        default=0.0,
        help="adam: decay of first order momentum of gradient",
    )
    parser.add_argument(
        "--beta2",
        type=float,
        default=0.9,
        help="adam: decay of first order momentum of gradient",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="number of cpu threads to use during batch generation",
    )
    parser.add_argument(
        "--latent_dim", type=int, default=128, help="dimensionality of the latent space"
    )
    parser.add_argument(
        "--img_size", type=int, default=32, help="size of each image dimension"
    )
    parser.add_argument(
        "--channels", type=int, default=3, help="number of image channels"
    )
    parser.add_argument(
        "--n_critic",
        type=int,
        default=1,
        help="number of training steps for discriminator per iter",
    )
    parser.add_argument(
        "--val_freq", type=int, default=20, help="interval between each validation"
    )
    parser.add_argument(
        "--print_freq", type=int, default=100, help="interval between each verbose"
    )
    parser.add_argument("--load_path", type=str, help="The reload model path")
    parser.add_argument("--log_path", type=str, default=None)
    parser.add_argument("--inception_path", type=str, default="./Inception_Net")

    parser.add_argument("--exp_name", type=str, help="The name of exp")
    parser.add_argument(
        "--d_spectral_norm",
        type=str2bool,
        default=False,
        help="add spectral_norm on discriminator?",
    )
    parser.add_argument(
        "--g_spectral_norm",
        type=str2bool,
        default=False,
        help="add spectral_norm on generator?",
    )
    parser.add_argument('--use_bn', '-use_bn', type=str2bool, default=False) # batch norm in discriminator
    parser.add_argument("--dataset", type=str, default="cifar10", help="dataset type")
    parser.add_argument(
        "--data_path", type=str, default="./data", help="The path of data set"
    )
    parser.add_argument(
        "--init_type",
        type=str,
        default="normal",
        choices=["normal", "orth", "xavier_uniform_", "false", "random"],
        help="The init type",
    )
    parser.add_argument(
        "--gf_dim", type=int, default=64, help="The base channel num of gen"
    )
    parser.add_argument(
        "--df_dim", type=int, default=64, help="The base channel num of disc"
    )
    parser.add_argument(
        "--gen_model", type=str, default="shared_gan", help="path of gen model"
    )
    parser.add_argument(
        "--model_type", choices=["random", "same"], default="random"
    )
    parser.add_argument(
        "--data_type",  choices=["random", "same"], default="random"
    )
    parser.add_argument(
        "--dis_model", type=str, default="shared_gan", help="path of dis model"
    )
    parser.add_argument(
        "--controller", type=str, default="controller", help="path of controller"
    )
    parser.add_argument("--eval_batch_size", type=int, default=200)
    parser.add_argument("--train_number", type=int, default=None)
    parser.add_argument("--num_eval_imgs", type=int, default=50000)
    parser.add_argument(
        "--bottom_width", type=int, default=4, help="the base resolution of the GAN"
    )
    parser.add_argument("--random_seed", type=int, default=12345)
    parser.add_argument("--Model_type", type=str, default="Forest_GAN", choices=["Forest_GAN", "MIX_GAN", "PAR_GAN"])

    # search
    parser.add_argument(
        "--shared_epoch",
        type=int,
        default=15,
        help="the number of epoch to train the shared gan at each search iteration",
    )
    parser.add_argument(
        "--grow_step1",
        type=int,
        default=25,
        help="which iteration to grow the image size from 8 to 16",
    )
    parser.add_argument(
        "--grow_step2",
        type=int,
        default=55,
        help="which iteration to grow the image size from 16 to 32",
    )
    parser.add_argument(
        "--max_search_iter",
        type=int,
        default=90,
        help="max search iterations of this algorithm",
    )
    parser.add_argument(
        "--ctrl_step",
        type=int,
        default=30,
        help="number of steps to train the controller at each search iteration",
    )
    parser.add_argument(
        "--ctrl_sample_batch",
        type=int,
        default=1,
        help="sample size of controller of each step",
    )
    parser.add_argument(
        "--hid_size", type=int, default=100, help="the size of hidden vector"
    )
    parser.add_argument(
        "--baseline_decay", type=float, default=0.9, help="baseline decay rate in RL"
    )
    parser.add_argument(
        "--rl_num_eval_img",
        type=int,
        default=5000,
        help="number of images to be sampled in order to get the reward",
    )
    parser.add_argument(
        "--num_candidate",
        type=int,
        default=10,
        help="number of candidate architectures to be sampled",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="preserve topk models architectures after each stage",
    )
    parser.add_argument(
        "--entropy_coeff", type=float, default=1e-3, help="to encourage the exploration"
    )
    parser.add_argument(
        "--dynamic_reset_threshold", type=float, default=1e-3, help="var threshold"
    )
    parser.add_argument(
        "--dynamic_reset_window", type=int, default=500, help="the window size"
    )
    parser.add_argument(
        "--arch", nargs="+", type=int, help="the vector of a discovered architecture"
    )
    parser.add_argument(
        "--num_disc", type=int, default=1
    )
    parser.add_argument(
        "--DiffAugment", default=None
    )
    parser.add_argument(
        "--train_mode", type=str, default="average", choices=["average", "softmax"]
    )
    parser.add_argument(
        "--random_state", type=int, default=2
    )
    parser.add_argument(
        "--probability", type=float, default=0.0
    )
    parser.add_argument(
        "--dataset_reduce", type=str2bool, default=False, help="the way of constructing bootstrap_bag"
    )
    parser.add_argument(
        "--loss", type=str, default="hinge_loss", help="target loss function in AutoGAN"
    )
    parser.add_argument(
        "--weight", type=int, default=5, help="target loss function in AutoGAN"
    )
    parser.add_argument(
        "--Aug_Flip", type=str2bool, default=False, help="Data Augment by flipping"
    )
    parser.add_argument(
        "--Diff_Aug", default="None", help="the operation of Differentiable Augmentatin"
    )

    # for distribution data parallel
    parser.add_argument("--local_rank", default=-1, type=int)

    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--distributed", type=bool, default=False)
    parser.add_argument("--dist_backend", type=str)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    return opt


