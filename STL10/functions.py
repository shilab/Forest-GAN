

import logging
import operator
import os
from copy import deepcopy

import time
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.autograd import Variable

from imageio import imsave
from torchvision.utils import make_grid
from tqdm import tqdm

from utils.fid_score import calculate_fid_given_paths
from utils.inception_score import get_inception_score
from utils.DiffAugment_pytorch import original, rand_brightness, rand_saturation, rand_contrast, rand_translation, rand_cutout


logger = logging.getLogger(__name__)


def train_shared(
    args,
    gen_net: nn.Module,
    dis_net: nn.Module,
    g_loss_history,
    d_loss_history,
    controller,
    gen_optimizer,
    dis_optimizer,
    train_loader,
    prev_hiddens=None,
    prev_archs=None,
):
    dynamic_reset = False
    logger.info("=> train shared GAN...")
    step = 0
    gen_step = 0

    # train mode
    gen_net.train()
    dis_net.train()

    # eval mode
    controller.eval()
    for epoch in range(args.shared_epoch):
        for iter_idx, (imgs, _) in enumerate(train_loader):

            # sample an arch
            arch = controller.sample(
                1, prev_hiddens=prev_hiddens, prev_archs=prev_archs
            )[0][0]
            gen_net.set_arch(arch, controller.cur_stage)
            dis_net.cur_stage = controller.cur_stage
            # Adversarial ground truths
            real_imgs = imgs.type(torch.cuda.FloatTensor)

            # Sample noise as generator input
            z = torch.cuda.FloatTensor(
                np.random.normal(0, 1, (imgs.shape[0], args.latent_dim))
            )

            # ---------------------
            #  Train Discriminator
            # ---------------------
            dis_optimizer.zero_grad()

            real_validity = dis_net(real_imgs)
            fake_imgs = gen_net(z).detach()
            assert fake_imgs.size() == real_imgs.size(), print(
                f"fake image size is {fake_imgs.size()}, "
                f"while real image size is {real_imgs.size()}"
            )

            fake_validity = dis_net(fake_imgs)

            # cal loss
            d_loss = torch.mean(
                nn.ReLU(inplace=True)(1.0 - real_validity)
            ) + torch.mean(nn.ReLU(inplace=True)(1 + fake_validity))
            d_loss.backward()
            dis_optimizer.step()

            # add to window
            d_loss_history.push(d_loss.item())

            # -----------------
            #  Train Generator
            # -----------------
            if step % args.n_critic == 0:
                gen_optimizer.zero_grad()

                gen_z = torch.cuda.FloatTensor(
                    np.random.normal(0, 1, (args.gen_batch_size, args.latent_dim))
                )
                gen_imgs = gen_net(gen_z)
                fake_validity = dis_net(gen_imgs)

                # cal loss
                g_loss = -torch.mean(fake_validity)
                g_loss.backward()
                gen_optimizer.step()

                # add to window
                g_loss_history.push(g_loss.item())
                gen_step += 1

            # verbose
            if gen_step and iter_idx % args.print_freq == 0:
                logger.info(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (
                        epoch,
                        args.shared_epoch,
                        iter_idx % len(train_loader),
                        len(train_loader),
                        d_loss.item(),
                        g_loss.item(),
                    )
                )

            # check window
            if g_loss_history.is_full():
                if (
                    g_loss_history.get_var() < args.dynamic_reset_threshold
                    or d_loss_history.get_var() < args.dynamic_reset_threshold
                ):
                    dynamic_reset = True
                    logger.info("=> dynamic resetting triggered")
                    g_loss_history.clear()
                    d_loss_history.clear()
                    return dynamic_reset

            step += 1

    return dynamic_reset


def train(
    args,
    gen_net: nn.Module,
    dis_net: nn.Module,     # List[nn.Model]
    gen_optimizer,
    dis_optimizer,
    gen_avg_param,
    train_loader,
    epoch,
    writer_dict,
    schedulers=None,
):
    writer = writer_dict["writer"]
    gen_step = 0

    # train mode
    """
        TODO-4
        Convert to training of multiple discriminators
    """
    gen_net = gen_net.train()
    dis_net = dis_net.train()

    for iter_idx, (imgs, _) in enumerate(tqdm(train_loader)):
        global_steps = writer_dict["train_global_steps"]

        # Adversarial ground truths
        real_imgs = imgs.type(torch.cuda.FloatTensor)

        # Sample noise as generator input
        z = torch.cuda.FloatTensor(
            np.random.normal(0, 1, (imgs.shape[0], args.latent_dim))
        )

        # ------------------------
        #   Train Discriminator
        # ------------------------
        dis_optimizer.zero_grad()

        real_validity = dis_net(real_imgs)
        fake_imgs = gen_net(z).detach()
        assert fake_imgs.size() == real_imgs.size()

        fake_validity = dis_net(fake_imgs)

        # cal loss
        d_loss = torch.mean(nn.ReLU(inplace=True)(1.0 - real_validity)) + torch.mean( nn.ReLU(inplace=True)(1 + fake_validity) )
        d_loss.backward()
        dis_optimizer.step()

        writer.add_scalar("d_loss", d_loss.item(), global_steps)

        # ---------------------
        #    Train Generator
        # ---------------------
        if global_steps % args.n_critic == 0:
            gen_optimizer.zero_grad()

            gen_z = torch.cuda.FloatTensor(
                np.random.normal(0, 1, (args.gen_batch_size, args.latent_dim))
            )
            gen_imgs = gen_net(gen_z)
            fake_validity = dis_net(gen_imgs)

            # cal loss
            g_loss = -torch.mean(fake_validity)
            g_loss.backward()
            gen_optimizer.step()

            # adjust learning rate
            if schedulers:
                gen_scheduler, dis_scheduler = schedulers
                g_lr = gen_scheduler.step(global_steps)
                d_lr = dis_scheduler.step(global_steps)
                writer.add_scalar("LR/g_lr", g_lr, global_steps)
                writer.add_scalar("LR/d_lr", d_lr, global_steps)

            # moving average weight
            for p, avg_p in zip(gen_net.parameters(), gen_avg_param):
                avg_p.mul_(0.999).add_(0.001, p.data)

            writer.add_scalar("g_loss", g_loss.item(), global_steps)
            gen_step += 1

        # verbose
        if gen_step and iter_idx % args.print_freq == 0:
            tqdm.write(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (
                    epoch,
                    args.max_epoch,
                    iter_idx % len(train_loader),
                    len(train_loader),
                    d_loss.item(),
                    g_loss.item(),
                )
            )
            
        writer_dict["train_global_steps"] = global_steps + 1


def gradient_penalty(dis_net, real_imgs, fake_imgs, device):
    alpha = torch.FloatTensor(np.random.random((real_imgs.size(0), 1, 1, 1))).to(device)
    interpolates = (alpha * real_imgs + ((1 - alpha) * fake_imgs)).requires_grad_(True)
    d_interpolates = dis_net(interpolates)

    fake = Variable(torch.FloatTensor(real_imgs.shape[0], 1).fill_(1.0), requires_grad=False).to(device)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]


    gradients = gradients.view(gradients.size(0), -1)
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gp


class CNN_UP_Augment(nn.Module):
    def __init__(self, seed):
        super(CNN_UP_Augment, self).__init__()

        self.conv2d = nn.Conv2d(3, 3, 3)
        self.activate = nn.Tanh()
        self.UpSample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, input):
        x = self.conv2d(input)
        x = self.activate(x)
        x = self.UpSample(x)

        return x


class CNN_Equal_Augment(nn.Module):
    def __init__(self, seed):
        super(CNN_Equal_Augment, self).__init__()

        self.conv2d = nn.Conv2d(3, 3, 3, bias=False, padding=1, )
        self.activate = nn.Tanh()

    def forward(self, input):
        x = self.conv2d(input)
        x = self.activate(x)

        return x

def Augment_Initialization(m):
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1:
        nn.init.orthogonal_(m.weight.data, 1.0)


def train_num_disc(
    args,
    gen_net,       # nn.Module
    disc_nets,     # list(nn.Model)
    gen_optimizer,
    disc_optimizers,
    gen_avg_param,
    train_loaders,
    epoch,
    writer_dict=None,
    schedulers=None,
    device=None
):
    if dist.get_rank() == 0:
        writer = writer_dict["writer"]
    
    AUGMENT_FNS = [original, rand_brightness, rand_saturation, rand_contrast, rand_translation, rand_cutout]
    CNN_UP_Augments = [CNN_UP_Augment(42+i).apply(Augment_Initialization).to(device) for i in range(6)]
    CNN_Equal_Augments = [CNN_Equal_Augment(96+i).apply(Augment_Initialization).to(device) for i in range(6)]

    # for mixed precision 
    # scaler = torch.cuda.amp.GradScaler() 
    # for mix 
    # train mode 
    """ 
        TODO-4 
        Convert to training of multiple discriminators 
    """ 

    start = time.time()
    for iter_idx, imgs, in enumerate(zip(*train_loaders)):
        pro = np.random.random((1,))[0]

        if dist.get_rank() == 0:
            global_steps = writer_dict["train_global_steps"]

        gen_step = 0
        # -------------------------
        #    Train Discriminator
        # -------------------------
        # Sample noise as generator input
        z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.dis_batch_size, args.latent_dim))).to(device)
        fake_imgs = gen_net(z).detach()
        d_loss_list = []
        for index in range(args.num_disc):
            dis_net = disc_nets[index]
            dis_net.train()

            disc_optimizers[index].zero_grad()

            if args.data_type == "random":
                real_imgs = imgs[index].type(torch.cuda.FloatTensor).to(device)
                z = torch.cuda.FloatTensor(np.random.normal(0, 1, (real_imgs.shape[0], args.latent_dim))).to(device)
                fake_imgs = gen_net(z).detach()
            elif args.data_type == "same":
                real_imgs = imgs[0].type(torch.cuda.FloatTensor).to(device)
            else:
                raise "data_type error!!!!!"
            
            if args.DiffAugment is not None:
                if args.DiffAugment == "CNN_UP_Aug": # Non_Aug
                    if args.num_disc == 1:
                        # import pdb; pdb.set_trace()
                        randn_index = np.random.randint(0, args.num_disc)
                        real_imgs = CNN_UP_Augments[randn_index](real_imgs)
                        fake_imgs = CNN_UP_Augments[randn_index](fake_imgs)
                    else:
                        real_imgs = CNN_UP_Augments[index](real_imgs)
                        fake_imgs = CNN_UP_Augments[index](fake_imgs)

                elif args.DiffAugment == "CNN_Equal_Aug":
                    if args.num_disc == 1:
                        # import pdb; pdb.set_trace()
                        randn_index = np.random.randint(0, args.num_disc)
                        real_imgs = CNN_Equal_Augments[randn_index](real_imgs)
                        fake_imgs = CNN_Equal_Augments[randn_index](fake_imgs)
                    else:
                        if pro < args.probability: # original
                            real_imgs = AUGMENT_FNS[0](real_imgs)
                            fake_imgs = AUGMENT_FNS[0](fake_imgs)
                        else:
                            real_imgs = CNN_Equal_Augments[index](real_imgs)
                            fake_imgs = CNN_Equal_Augments[index](fake_imgs)

                else: # default DiffAugment; Linear
                    if args.num_disc == 1:
                        randn_index = np.random.randint(0, args.num_disc)
                        real_imgs = AUGMENT_FNS[randn_index](real_imgs)
                        fake_imgs = AUGMENT_FNS[randn_index](fake_imgs)
                    else:
                        real_imgs = AUGMENT_FNS[index](real_imgs)
                        fake_imgs = AUGMENT_FNS[index](fake_imgs)

            # Diffeneratiable Augmentation 
            if not args.Aug_Flip and ("None" not in args.Diff_Aug):
                from utils.diff_aug import DiffAugment
                real_imgs = DiffAugment(real_imgs, args.Diff_Aug, True)
                fake_imgs = DiffAugment(fake_imgs, args.Diff_Aug, True)
            
            real_validity = dis_net(real_imgs)
            fake_validity = dis_net(fake_imgs)

            # purely from the idea of hinge loss
            if args.loss == "wgan_gp_loss":
                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity)
                gp = gradient_penalty(dis_net, real_imgs, fake_imgs, device)
                d_loss = d_loss + args.weight * gp
                d_loss_list.append(d_loss.item())

                d_loss.backward()
                disc_optimizers[index].step()

            elif args.loss == "vanilla_loss":
                real_validity = nn.Sigmoid()(real_validity)
                fake_validity = nn.Sigmoid()(fake_validity)

                real = torch.ones_like(real_validity).to(device)
                fake = torch.zeros_like(fake_validity).to(device)

                # fake = torch.FloatTensor(fake_validity.size(0), 1).fill_(0.0).to(device)

                d_loss = torch.nn.BCELoss()(real_validity, real) + torch.nn.BCELoss()(fake_validity, fake)
                d_loss_list.append(d_loss.item())
                
                d_loss.backward()
                disc_optimizers[index].step()

            else:   # default hinge-loss

                # # amended loss function
                # real_validity = 1.0 - real_validity
                # mask = torch.ones(real_validity.size()).to(device)
                # mask = (real_validity > 0)*mask
                # real_validity = real_validity * mask

                # fake_validity = 1.0 + fake_validity
                # mask = torch.ones(fake_validity.size()).to(device)
                # mask = (fake_validity > 0)*mask
                # fake_validity = fake_validity * mask

                # d_loss = torch.mean(nn.ReLU(inplace=True)(real_validity)) + torch.mean( nn.ReLU(inplace=True)(fake_validity))

                # the hinge loss in AutoGAN
                d_loss = torch.mean(nn.ReLU(inplace=True)(1.0 - real_validity)) + torch.mean( nn.ReLU(inplace=True)(1 + fake_validity))
                d_loss_list.append(d_loss.item())

                d_loss.backward()
                disc_optimizers[index].step()

            # # mixed precision
            # with torch.autocast(device_type="cuda", dtype=torch.float16):
            #     fake_imgs = gen_net(z).detach()
            #     real_validity = dis_net(real_imgs)
            #     fake_validity = dis_net(fake_imgs)

            #     # cal loss
            #     d_loss = torch.mean(nn.ReLU(inplace=True)(1.0 - real_validity)) + torch.mean( nn.ReLU(inplace=True)(1 + fake_validity) )
            #     d_loss_list.append(d_loss)

            # scaler.scale(d_loss).backward()
            # scaler.step(disc_optimizers[index])
            # scaler.update()
            
            if dist.get_rank() == 0:
                writer.add_scalar("disc-{}_loss".format(index), d_loss.item(), global_steps)
        
        # --------------------
        #   Train Generator 
        # --------------------
        
        if iter_idx % args.n_critic == 0:
            # zero the parameter gradients
            gen_net.train()
            gen_optimizer.zero_grad()

            avg_loss = 0.0
            for index in range(args.num_disc):
                gen_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.gen_batch_size, args.latent_dim)))
                fake_imgs = gen_net(gen_z)

                # Differentiable Augmentation
                if args.DiffAugment is not None:
                    if args.DiffAugment == "CNN_UP_Aug":
                        if args.num_disc == 1:
                            randn_index = np.random.randint(0, 5)
                            fake_imgs = CNN_UP_Augments[randn_index](fake_imgs)
                        else:
                            fake_imgs = CNN_UP_Augments[index](fake_imgs)
                    
                    elif args.DiffAugment == "CNN_Equal_Aug":
                        a = 1
                        if args.num_disc == 1:
                            randn_index = np.random.randint(0, 5)
                            fake_imgs = CNN_Equal_Augments[randn_index](fake_imgs)
                        else:
                            fake_imgs = CNN_Equal_Augments[index](fake_imgs)
                    else:
                        if args.num_disc == 1:
                            randn_index = np.random.randint(0, 5)
                            fake_imgs = AUGMENT_FNS[randn_index](fake_imgs)
                        else:
                            fake_imgs = AUGMENT_FNS[index](fake_imgs)

                # Diffeneratiable Augmentation 
                if not args.Aug_Flip and ("None" not in args.Diff_Aug):
                    from utils.diff_aug import DiffAugment
                    fake_imgs = DiffAugment(fake_imgs, args.Diff_Aug, True)
                fake_validity = disc_nets[index](fake_imgs)

                # cal loss 
                if args.loss == "wgan_gp_loss":
                    g_loss = -torch.mean(fake_validity) * (1/args.num_disc) 

                elif args.loss == "vanilla_loss":
                    fake_validity = torch.nn.Sigmoid()(fake_validity)
                    real = Variable(torch.FloatTensor(fake_validity.size(0), 1).fill_(1.0), requires_grad=False).to(device)
                    g_loss = torch.nn.BCELoss()(fake_validity, real) * (1/args.num_disc) 

                else: # default hinge loss 
                    g_loss = -torch.mean(fake_validity) * (1/args.num_disc) 
                
                g_loss.backward()
                avg_loss += g_loss

            gen_optimizer.step()

            # adjust learning rate
            if schedulers:
                gen_scheduler, disc_schedulers = schedulers
                g_lr = gen_scheduler.step(global_steps)
                if dist.get_rank() == 0:
                    writer.add_scalar("LR/g_lr", g_lr, global_steps)
                for i in range(args.num_disc):
                    d_lr = disc_schedulers[i].step(global_steps)
                    if dist.get_rank() == 0:
                        writer.add_scalar("LR/d-{}_lr".format(i), d_lr, global_steps)

            # moving average weight
            for p, avg_p in zip(gen_net.parameters(), gen_avg_param):
                avg_p.mul_(0.999).add_(0.001, p.data)

            if dist.get_rank() == 0:
                writer.add_scalar("g_loss", avg_loss.item(), global_steps)
                gen_step += 1

        # verbose
        if dist.get_rank() == 0:
            if gen_step and iter_idx % args.print_freq == 0:
                end = time.time()
                print("[Epoch {}/{}] [Batch {}/{}]; G loss: {:.7f}; D loss: {}; time: {:.5f}".format(
                                                                                    epoch,
                                                                                    args.max_epoch,
                                                                                    iter_idx % len(train_loaders[0]),
                                                                                    len(train_loaders[0]),
                                                                                    avg_loss.item(),
                                                                                    [loss_ for loss_ in d_loss_list],
                                                                                    end-start
                                                                                ))
                start = time.time()
            writer_dict["train_global_steps"] = global_steps + 1


def train_controller(args, controller, ctrl_optimizer, gen_net, prev_hiddens, prev_archs, writer_dict):
    logger.info("=> train controller...")
    writer = writer_dict["writer"]
    baseline = None

    # train mode
    controller.train()

    # eval mode
    gen_net.eval()

    cur_stage = controller.cur_stage
    for step in range(args.ctrl_step):
        controller_step = writer_dict["controller_steps"]
        archs, selected_log_probs, entropies = controller.sample(
            args.ctrl_sample_batch, prev_hiddens=prev_hiddens, prev_archs=prev_archs
        )
        cur_batch_rewards = []
        for arch in archs:
            logger.info(f"arch: {arch}")
            gen_net.set_arch(arch, cur_stage)
            is_score = get_is(args, gen_net, args.rl_num_eval_img)
            logger.info(f"get Inception score of {is_score}")
            cur_batch_rewards.append(is_score)
        cur_batch_rewards = torch.tensor(cur_batch_rewards, requires_grad=False).cuda()
        cur_batch_rewards = (
            cur_batch_rewards.unsqueeze(-1) + args.entropy_coeff * entropies
        )  # bs * 1
        if baseline is None:
            baseline = cur_batch_rewards
        else:
            baseline = (
                args.baseline_decay * baseline.detach()
                + (1 - args.baseline_decay) * cur_batch_rewards
            )
        adv = cur_batch_rewards - baseline

        # policy loss
        loss = -selected_log_probs * adv
        loss = loss.sum()

        # update controller
        ctrl_optimizer.zero_grad()
        loss.backward()
        ctrl_optimizer.step()

        # write
        mean_reward = cur_batch_rewards.mean().item()
        mean_adv = adv.mean().item()
        mean_entropy = entropies.mean().item()
        writer.add_scalar("controller/loss", loss.item(), controller_step)
        writer.add_scalar("controller/reward", mean_reward, controller_step)
        writer.add_scalar("controller/entropy", mean_entropy, controller_step)
        writer.add_scalar("controller/adv", mean_adv, controller_step)

        writer_dict["controller_steps"] = controller_step + 1


def get_is(args, gen_net: nn.Module, num_img):
    """
    Get inception score.
    :param args:
    :param gen_net:
    :param num_img:
    :return: Inception score
    """

    # eval mode
    gen_net = gen_net.eval()

    eval_iter = num_img // args.eval_batch_size
    img_list = list()
    for _ in range(eval_iter):
        z = torch.cuda.FloatTensor(
            np.random.normal(0, 1, (args.eval_batch_size, args.latent_dim))
        )

        # Generate a batch of images
        gen_imgs = (
            gen_net(z)
            .mul_(127.5)
            .add_(127.5)
            .clamp_(0.0, 255.0)
            .permute(0, 2, 3, 1)
            .to("cpu", torch.uint8)
            .numpy()
        )
        img_list.extend(list(gen_imgs))

    # get inception score
    logger.info("calculate Inception score...")
    mean, std = get_inception_score(args, img_list)

    return mean


def Generate_images(args, epoch, gen_net: nn.Module):

    # eval mode
    gen_net = gen_net.eval()

    # get fid and inception score
    if dist.get_rank() == 0:
        fid_buffer_dir = os.path.join(args.path_helper["sample_path"], "fid_buffer-epoch={}".format(epoch))

        os.makedirs(fid_buffer_dir, exist_ok=True)

    eval_iter = args.num_eval_imgs // args.eval_batch_size
    for iter_idx in range(eval_iter):
        z = torch.cuda.FloatTensor(
            np.random.normal(0, 1, (args.eval_batch_size, args.latent_dim))
        )

        # Generate a batch of images
        gen_imgs = (
            gen_net(z)
            .mul_(127.5)
            .add_(127.5)
            .clamp_(0.0, 255.0)
            .permute(0, 2, 3, 1)
            .to("cpu", torch.uint8)
            .numpy()
        )
        
        if dist.get_rank() == 0:
            
            for img_idx, img in enumerate(gen_imgs):
                file_name = os.path.join(fid_buffer_dir, f"iter{iter_idx}_b{img_idx}.png")
                imsave(file_name, img)


def validate(args, epoch, fixed_z, fid_stat, gen_net: nn.Module, writer_dict, clean_dir=True):
    writer = writer_dict["writer"]
    global_steps = writer_dict["valid_global_steps"]

    # eval mode
    gen_net = gen_net.eval()

    # generate images
    sample_imgs = gen_net(fixed_z)
    img_grid = make_grid(sample_imgs, nrow=5, normalize=True, scale_each=True)

    # get fid and inception score
    fid_buffer_dir = os.path.join(args.path_helper["sample_path"], "fid_buffer-epoch={}".format(epoch))
    os.makedirs(fid_buffer_dir, exist_ok=True)

    eval_iter = args.num_eval_imgs // (args.eval_batch_size)
    img_list = list()
    logger.info("=> epoch: {}; generate fake images...".format(epoch))
    for iter_idx in range(eval_iter):
        z = torch.cuda.FloatTensor(
            np.random.normal(0, 1, (args.eval_batch_size, args.latent_dim))
        )

        # Generate a batch of images
        gen_imgs = (
            gen_net(z)
            .mul_(127.5)
            .add_(127.5)
            .clamp_(0.0, 255.0)
            .permute(0, 2, 3, 1)
            .to("cpu", torch.uint8)
            .numpy()
        )
        
        for img_idx, img in enumerate(gen_imgs):
            file_name = os.path.join(fid_buffer_dir, f"iter{iter_idx}_b{img_idx}.png")
            imsave(file_name, img)
        img_list.extend(list(gen_imgs))

    # get inception score
    print("=> epoch: {}; calculate inception score...".format(epoch))
    mean, std = get_inception_score(args, img_list)
    print(f"Inception score: {mean}")

    # get fid score
    print("=> epoch: {}; calculate fid score...".format(epoch))
    fid_score = calculate_fid_given_paths(
        [fid_buffer_dir, fid_stat], inception_path=args.inception_path
    )

    print(f"FID score: {fid_score}")

    # if clean_dir:
    #     os.system("rm -r {}".format(fid_buffer_dir))
    # else:
    #     logger.info(f"=> sampled images are saved to {fid_buffer_dir}")
    # writer.add_image("sampled_images", img_grid, global_steps)
    # writer.add_scalar("Inception_score/std", std, global_steps)

    writer.add_scalar("Inception_score/mean", mean, global_steps)
    writer.add_scalar("FID_score", fid_score, global_steps)

    writer_dict["valid_global_steps"] = global_steps + 1

    return mean, fid_score


def get_topk_arch_hidden(args, controller, gen_net, prev_archs, prev_hiddens):
    """
    ~
    :param args:
    :param controller:
    :param gen_net:
    :param prev_archs: previous architecture
    :param prev_hiddens: previous hidden vector
    :return: a list of topk archs and hiddens.
    """
    logger.info(
        f"=> get top{args.topk} archs out of {args.num_candidate} candidate archs..."
    )
    assert args.num_candidate >= args.topk
    controller.eval()
    cur_stage = controller.cur_stage
    archs, _, _, hiddens = controller.sample(
        args.num_candidate,
        with_hidden=True,
        prev_archs=prev_archs,
        prev_hiddens=prev_hiddens,
    )
    hxs, cxs = hiddens 
    arch_idx_perf_table = {}
    for arch_idx in range(len(archs)):
        logger.info(f"arch: {archs[arch_idx]}")
        gen_net.set_arch(archs[arch_idx], cur_stage)
        is_score = get_is(args, gen_net, args.rl_num_eval_img)
        logger.info(f"get Inception score of {is_score}")
        arch_idx_perf_table[arch_idx] = is_score
    topk_arch_idx_perf = sorted(
        arch_idx_perf_table.items(), key=operator.itemgetter(1)
    )[::-1][: args.topk]
    topk_archs = []
    topk_hxs = []
    topk_cxs = []
    logger.info(f"top{args.topk} archs:")
    for arch_idx_perf in topk_arch_idx_perf:
        logger.info(arch_idx_perf)
        arch_idx = arch_idx_perf[0]
        topk_archs.append(archs[arch_idx])
        topk_hxs.append(hxs[arch_idx].detach().requires_grad_(False))
        topk_cxs.append(cxs[arch_idx].detach().requires_grad_(False))

    return topk_archs, (topk_hxs, topk_cxs)


class LinearLrDecay(object):
    def __init__(self, optimizer, start_lr, end_lr, decay_start_step, decay_end_step):

        assert start_lr > end_lr
        self.optimizer = optimizer
        self.delta = (start_lr - end_lr) / (decay_end_step - decay_start_step)
        self.decay_start_step = decay_start_step
        self.decay_end_step = decay_end_step
        self.start_lr = start_lr
        self.end_lr = end_lr

    def step(self, current_step):
        if current_step <= self.decay_start_step:
            lr = self.start_lr
        elif current_step >= self.decay_end_step:
            lr = self.end_lr
        else:
            lr = self.start_lr - self.delta * (current_step - self.decay_start_step)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr
        return lr


def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)


def copy_params(model):
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten
