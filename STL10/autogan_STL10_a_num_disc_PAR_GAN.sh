#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node 2  --master_port 56563 train_myself.py \
-gen_bs 192 \
-dis_bs 96 \
--dataset stl10 \
--bottom_width 6 \
--img_size 48 \
--max_iter 300000 \
--num_eval_imgs 100000 \
--gen_model autogan_cifar10_a \
--dis_model autogan_cifar10_a \
--latent_dim 128 \
--gf_dim 256 \
--df_dim 128 \
--g_spectral_norm False \
--d_spectral_norm True \
--g_lr 0.00025 \
--d_lr 0.00025 \
--beta1 0.0 \
--beta2 0.9 \
--init_type xavier_uniform_ \
--n_critic 5 \
--val_freq 20 \
--exp_name autogan_stl10_a \
--num_disc 10 \
--random_state 42 \
--num_workers 0 \
--loss hinge_loss \
--Model_type PAR_GAN

