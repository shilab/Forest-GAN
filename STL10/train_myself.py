
from __future__ import absolute_import, division, print_function

import os
from copy import deepcopy

import time
import torch
import numpy as np
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tensorboardX import SummaryWriter


import cfg
import datasets
import models  # noqa
from functions import copy_params, LinearLrDecay, load_params, train, train_num_disc, validate, Generate_images
from utils.fid_score import check_or_download_inception, create_inception_graph
# from utils.inception_score import _init_inception
from utils.utils import create_logger, save_checkpoint, set_log_dir

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def main():
    args = cfg.parse_args()

    local_rank = args.local_rank
    # DDP backend initialize
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    device = torch.device("cuda", local_rank)

    torch.cuda.manual_seed(args.random_seed)

    def weights_init_xavier_uniform(m):
        classname = m.__class__.__name__
        if classname.find("Conv2d") != -1 or type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight.data, 1.0)
    
    def weights_init_xavier_normal(m):
        classname = m.__class__.__name__
        if classname.find("Conv2d") != -1 or type(m) == nn.Linear:
            nn.init.xavier_normal_(m.weight.data, 1.0)

    def weights_init_kaiming_uniform(m):
        classname = m.__class__.__name__
        if classname.find("Conv2d") != -1:
            nn.init.kaiming_uniform_(m.weight.data, mode='fan_out', nonlinearity='relu')
        elif type(m) == nn.Linear:
            nn.init.kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='relu')

    def weights_init_kaiming_normal(m):
        classname = m.__class__.__name__
        if classname.find("Conv2d") != -1:
            nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
        elif type(m) == nn.Linear:
            nn.init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='relu')

    def weights_init_orthogonal(m):
        classname = m.__class__.__name__
        if classname.find("Conv2d") != -1 or type(m) == nn.Linear:
            nn.init.orthogonal_(m.weight.data, 1.0)

    # # original initialization in AutoGAN
    # # weight init
    # def weights_init(m):
    #     classname = m.__class__.__name__
    #     if classname.find("Conv2d") != -1:
    #         if args.init_type == "normal":
    #             nn.init.normal_(m.weight.data, 0.0, 0.02)
    #         elif args.init_type == "orth":
    #             nn.init.orthogonal_(m.weight.data)
    #         elif args.init_type == "xavier_uniform_":
    #             nn.init.xavier_uniform_(m.weight.data, 1.0)
    #         else:
    #             raise NotImplementedError(
    #                 "{} unknown inital type".format(args.init_type)
    #             )
    #     elif classname.find("BatchNorm2d") != -1:
    #         nn.init.normal_(m.weight.data, 1.0, 0.02)
    #         nn.init.constant_(m.bias.data, 0.0)
    
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        # random.seed(seed)
        torch.backends.cudnn.deterministic = True
    """
        TODO-1
        add multiple discriminators
    """
    # import network
    gen_net = eval("models." + args.gen_model + ".Generator")(args=args).to(device)
    gen_net.apply(weights_init_xavier_uniform)
    gen_net = DDP(gen_net, device_ids=[local_rank], output_device=local_rank)

    disc_nets = []
    disc_net_initialization_list = [weights_init_xavier_uniform, weights_init_xavier_normal, weights_init_kaiming_uniform, weights_init_kaiming_normal, weights_init_orthogonal]
    for disc_index in range(args.num_disc):
        # random seed
        if args.model_type == "random":
            seed = args.random_state + disc_index
        elif args.model_type == "same":
            seed = args.random_state
        else:
            raise "model_type error!!!!!"
        setup_seed(seed)

        disc_net = eval("models." + args.dis_model + ".Discriminator")(args=args).to(device)
        if args.init_type != "random":
            disc_index_ini = 0
        else:
            disc_index_ini = disc_index
        disc_net.apply(disc_net_initialization_list[disc_index_ini])
        
        disc_net = DDP(disc_net, device_ids=[local_rank], output_device=local_rank)
        disc_nets.append(disc_net)        

    # set optimizer
    gen_optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, gen_net.parameters()),
        args.g_lr,
        (args.beta1, args.beta2),
    )

    disc_optimizers = []
    for index in range(args.num_disc):
        disc_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, disc_nets[index].parameters()),
            args.d_lr,
            (args.beta1, args.beta2),
        )
        disc_optimizers.append(disc_optimizer)

    gen_scheduler = LinearLrDecay(
        gen_optimizer, args.g_lr, args.g_lr*0.0001, args.max_iter * 0.25, args.max_iter
    )

    disc_schedulers = []
    for index in range(args.num_disc):
        disc_scheduler = LinearLrDecay(
            disc_optimizers[index], args.d_lr, args.d_lr*0.0001, args.max_iter * 0.25, args.max_iter
        )
        disc_schedulers.append(disc_scheduler)

    # finished DIstributedSAampler
    # for each train_loader in train_loaders, it satisfy the format of bootstrap-bag
    args.dis_batch_size = args.dis_batch_size // 3
    args.gen_batch_size = args.gen_batch_size // 3
    dataset = datasets.ImageDataset_bootstrap_bags(args)
    train_loaders = dataset.trains
    
    # epoch number for dis_net
    args.max_epoch = args.max_epoch * args.n_critic
    if args.max_iter:
        args.max_epoch = np.ceil(args.max_iter * args.n_critic / len(train_loaders[0])) # default n_critic=5

    # initial
    fixed_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (25, args.latent_dim)))
    gen_avg_param = copy_params(gen_net)
    start_epoch = 0
    best_fid = 1e4

    if args.load_path:
        print("------- loading model -------")
        args.load_path = os.path.join("logs", args.load_path)
        checkpoint_file = os.path.join(args.load_path, "Model", "checkpoint-epoch=1430.pth")
        map_location = {'cuda:%d' % 0: 'cuda:%d' % dist.get_rank()}
        checkpoint = torch.load(checkpoint_file, map_location=map_location)

        start_epoch = checkpoint["epoch"]
        best_fid = checkpoint["best_fid"]

        gen_net.load_state_dict(checkpoint["gen_state_dict"])
        gen_optimizer.load_state_dict(checkpoint["gen_optimizer"])

        for index in range(args.num_disc):
            disc_nets[index].load_state_dict(checkpoint["dic_state_dicts"][index])
            disc_optimizers[index].load_state_dict(checkpoint["dis_optimizers"][index])

        avg_gen_net = deepcopy(gen_net)
        avg_gen_net.load_state_dict(checkpoint["avg_gen_state_dict"])
        gen_avg_param = copy_params(avg_gen_net)
        del avg_gen_net

        args.path_helper = checkpoint["path_helper"]

        if dist.get_rank() == 0:
            logger = create_logger(args.path_helper["log_path"])
            logger.info(f"=> loaded checkpoint {checkpoint_file} (epoch {start_epoch})")
            print("load model successively.")
    
    else:
        if dist.get_rank() == 0:
            # create new log dir
            assert args.exp_name
            logs = "logs"
            if args.log_path is not None:
                logs = args.log_path
            args.path_helper = set_log_dir(logs, args.exp_name)
            logger = create_logger(args.path_helper["log_path"])
    
    if dist.get_rank() == 0:
        logger.info(args)

        writer_dict = {
                "writer": SummaryWriter(args.path_helper["log_path"]),
                "train_global_steps": start_epoch * len(train_loaders[0]),
                "valid_global_steps": start_epoch // args.val_freq,
        }
    else:
        writer_dict = None
    
    # train loop
    lr_schedulers = (gen_scheduler, disc_schedulers) if args.lr_decay else None
    if dist.get_rank() == 0:
        if args.lr_decay:
            print("Result: {}".format(args.lr_decay))
        else:
            print("is none")

    for epoch in range(int(start_epoch), int(args.max_epoch), 1):
        if dist.get_rank() == 0:
            start = time.time()
            print("epoch: {}".format(epoch))

        train_num_disc(
            args,
            gen_net,
            disc_nets,
            gen_optimizer,
            disc_optimizers,
            gen_avg_param,
            train_loaders,
            epoch,
            writer_dict,
            lr_schedulers,
            device
        )

        if epoch % 10 == 0:
                
            if dist.get_rank() == 0:
                avg_gen_net = deepcopy(gen_net)
                load_params(avg_gen_net, gen_avg_param)
                save_checkpoint(
                    {
                            "epoch": epoch + 1,
                            "gen_model": args.gen_model,
                            "dis_model": args.dis_model,
                            "gen_state_dict": gen_net.state_dict(),
                            "dic_state_dicts": [disc_nets[i].state_dict() for i in range(args.num_disc)], 
                            "avg_gen_state_dict": avg_gen_net.state_dict(),
                            "gen_optimizer": gen_optimizer.state_dict(),
                            "dis_optimizers": [disc_optimizers[i].state_dict() for i in range(args.num_disc)],
                            "best_fid": best_fid,
                            "path_helper": args.path_helper,
                        },
                        os.path.join(args.path_helper["ckpt_path"], "checkpoint-epoch={}.pth".format(epoch)),
                    )
                del avg_gen_net

            dist.barrier() # make sure that the main process loads the model after process

        if dist.get_rank() == 0:
            end = time.time()
            print("[{}]-epoch={}; Time: {:.5f}".format(args.loss, epoch, end-start))

        if epoch > 2000:
            break
            
if __name__ == "__main__":
    main()
