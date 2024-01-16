import os
import time
import math
import pathlib
import argparse
import numpy as np
from imageio.v2 import imread
import torch
import tensorflow.compat.v1 as tf   # temsorflow2.0

import matplotlib.pyplot as plt

from tensorboardX import SummaryWriter

tf.disable_eager_execution()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

from evaluation_util import evaluation_precision_recall
import cfg_gen_evl
import models 
import sys
# sys.path.append("..")
from utils.fid_score import calculate_fid_given_paths, calculate_frechet_distance
from utils.inception_score import get_inception_score, _init_inception

# os.environ["CUDA_VISIBLE_DEVICES"] = "6"

def generate_image(args, gen_net, checkpoint_file):

    checkpoint = torch.load(checkpoint_file)
    gene_state_dict = dict()
    for key, value in checkpoint["avg_gen_state_dict"].items():
        if "module." in key:
            gene_state_dict[key[7:]] = value
        else:
            gene_state_dict[key] = value

    gen_net.load_state_dict(gene_state_dict)
    args.path_helper = checkpoint["path_helper"]

    gen_net.eval()
    eval_iter = args.num_eval_imgs // args.batch_size
    result_images = []
    for iter_idx in range(eval_iter):
        z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.batch_size, args.latent_dim)))

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
        
        if len(result_images) == 0:
            result_images = gen_imgs
        else:
            result_images = np.concatenate((result_images, gen_imgs), axis=0)

    return result_images

if __name__ == "__main__":
    
    args = cfg_gen_evl.parse_args()
    args.log_dir = "logs/" + args.log_dir

    log_file_fid = args.log_dir + "/FID_IS_2.txt"
    log_fid = open(log_file_fid, "w")
    content = "Epoch\tFID\tIS\n"
    log_fid.write(content)
    log_fid.flush()

    log_file_pr = args.log_dir + "/Pre_Rec_2.txt"
    log_pr = open(log_file_pr, "w")
    content = "Epoch\tPrecision\tRecall\n"
    log_pr.write(content)
    log_pr.flush()

    IS_Score_list, FID_Score_list = [], []
    precision_list, recall_list = [], []
    summarywriter = SummaryWriter(args.log_dir + "/Log")
    
    # initialize the Inception graph
    _init_inception(args)
    with tf.Session(config=config) as sess:
        pool3 = sess.graph.get_tensor_by_name("pool_3:0")     # pool3 is for the calculation of the metric of FID, precision and recall
        ops = pool3.graph.get_operations()
        for op_idx, op in enumerate(ops):
            for o in op.outputs:
                shape = o.get_shape()
                if shape._dims != []:
                    shape = [s for s in shape]
                    new_shape = []
                    for j, s in enumerate(shape):
                        if s == 1 and j == 0:
                            new_shape.append(None)
                        else:
                            new_shape.append(s)
                    o.__dict__["_shape_val"] = tf.TensorShape(new_shape)
        w = sess.graph.get_operation_by_name("softmax/logits/MatMul").inputs[1]
        logits = tf.matmul(tf.squeeze(pool3, [1, 2]), w)

        softmax = tf.nn.softmax(logits)                       # softmax is for the calculation of the metric of IS

    # generate the mean and sigma features for ral images
    # file_real = np.load(args.fid_stat)
    # real_images_mean, real_images_sigma = file_real['mu'][:], file_real['sigma'][:]     # the mean and sigma for real images
    # file_real.close()

    # loading real image; and calculate their Inception features for the calculation of precision and recall
    real_images_path = args.real_images_path        # [50000, 32, 32, 3] or [100000, 48, 48, 3], pixel_value=[-1, 1]
    real_images = np.load(real_images_path)
    print("real images: {}".format(real_images.shape))
    # import pdb; pdb.set_trace()
    real_images = np.clip(real_images * 127.5 + 127.5, 0.0, 255.0)

    # generate real features from real-features, which can be used in the calculation of precision and recall
    batch_size = args.batch_size
    real_images_len = len(real_images)
    if args.numbers is None:
        numbers = real_images_len
    else:
        numbers = args.numbers
    
    real_features = np.empty((numbers, 2048))
    n_batches = int(math.ceil(float(numbers) / float(batch_size)))

    with tf.Session(config=config) as sess:
        for i in range(n_batches):
            images = real_images[(i * batch_size): min((i+1)*batch_size, numbers)]
            features = sess.run(pool3, {"ExpandDims:0": images}) 
            
            start = i * batch_size
            end = min((i+1)*batch_size, numbers)
            # print(features.shape())
            # import pdb; pdb.set_trace()
            real_features[start:end] = features.reshape(end-start, -1)

    print("real feature: {}".format(real_features.shape))
    # for calculation of FID
    real_images_mean = np.mean(real_features, axis=0)
    real_images_sigma = np.cov(real_features, rowvar=False)
    
    # generate features for fake fakes of each epoch
    # and the calculate the metric of FID, IS, precision and recall
    # define the generator
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gen_net = eval("models." + args.gen_model + ".Generator")(args=args).to(device)
    args.num_eval_imgs = numbers
    
    for epoch in range(args.start_epoch, args.epoches, 1):
    
        checkpoint_file = os.path.join("logs", args.load_path, "Model", "checkpoint-epoch={}.pth".format(epoch))
        if not os.path.exists(checkpoint_file):
            continue
        
        start_time = time.time()

        predictions = []
        splits = args.splits
        fake_images = generate_image(args, gen_net, checkpoint_file)
        fake_features = np.empty((len(fake_images), 2048))
        n_batches = len(fake_images) // batch_size

        plt.figure(figsize=(40, 4))
        for i in range(10):
            plt.subplot(2, 10, i+1)
            plt.imshow((fake_images[i])/255.0, cmap='gray')

            plt.subplot(2, 10, i+11)
            plt.imshow((real_images[epoch*5+i])/255.0, cmap='gray')

            plt.axis("off")

        plt.savefig("logs/{}/Samples/epoch={}.png".format(args.load_path, epoch), transparent=True, bbox_inches='tight')
        plt.close()

        # np.save("logs/{}/Samples/epoch={}.npy".format(args.load_path, epoch), fake_images)

        print("fake images: {}".format(fake_images.shape))
        
        with tf.Session(config=config) as sess:
            # real the generated images
            for i in range(n_batches):

                sys.stdout.flush()
                images = fake_images[(i * batch_size): min((i+1)*batch_size, len(fake_images))]

                prediction, features = sess.run([softmax, pool3], {"ExpandDims:0": images}) 
                predictions.append(prediction)
                start = i * batch_size
                end = min((i+1)*batch_size, len(fake_images))
                
                fake_features[start:end] = features.reshape(end-start, -1)
            
            # for IS_score
            predictions = np.concatenate(predictions, 0)
            scores = []
            for i in range(splits): 
                part = predictions[(i * predictions.shape[0] // splits) : ((i + 1) * predictions.shape[0] // splits), :]
                kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
                kl = np.mean(np.sum(kl, 1))
                scores.append(np.exp(kl))

            IS_score = np.mean(scores)

            # for FID_score 
            fake_images_mean = np.mean(fake_features, axis=0)
            fake_images_sigma = np.cov(fake_features, rowvar=False)
            FID_score = calculate_frechet_distance(real_images_mean, real_images_sigma, fake_images_mean, fake_images_sigma)

            # calculate precision and recall
            # real_num_samples = min(len(real_features), 50000)
            # fake_num_samples = min(len(fake_features), 50000)
            # real_indexes = np.random.choice(len(real_features), real_num_samples, replace=False)
            # fake_indexes = np.random.choice(len(fake_features), fake_num_samples, replace=False)
            # precision, recall = evaluation_precision_recall(real_features[real_indexes], fake_features[fake_indexes], args.lambd, args.topk)

            precision, recall = 0.0, 0.0

            summarywriter.add_scalar("IS_Score", IS_score, epoch+1)
            summarywriter.add_scalar("FID_Score", FID_score, epoch+1)
            summarywriter.add_scalar("precision", precision, epoch+1)
            summarywriter.add_scalar("recall", recall, epoch+1)

            IS_Score_list.append(IS_score)
            FID_Score_list.append(FID_score)
            precision_list.append(precision)
            recall_list.append(recall)

            end_time = time.time()

            print("epoch: {:4d}, is_score: {:.5f}, fid_score: {:.5f}, pre: {:.5f}, rec: {:.5f}; time: {:.5f}".format(\
                epoch, IS_score, FID_score, precision, recall, end_time-start_time))

            content = "{:3d}\t{:.5f}\t{:.5f}\n".format(epoch, FID_score, IS_score)
            log_fid.write(content)
            log_fid.flush()

            content = "{:3d}\t{:.5f}\t{:.5f}\n".format(epoch, precision, recall)
            log_pr.write(content)
            log_pr.flush()


"""
if __name__ == "__main__":
    log_dir = dict()
    log_dir[1] = "autogan_cifar10_a_2023_03_01_23_00_19"
    log_dir[2] = "autogan_cifar10_a_2023_03_01_23_00_51"
    log_dir[5] = "autogan_cifar10_a_2023_03_01_23_01_23"

    args = parse_args()
    
    log_file = log_dir[args.num_disc] + "/FID_IS.txt"
    log = open(log_file, "w")
    content = "FID\tIS\n"
    log.write(content)
    log.flush()

    IS_Score_list, FID_Score_list = [], []
    # summarywriter = SummaryWriter(log_dir[args.num_disc] + "/Log")

    # initialize the graph
    _init_inception(args)

    for epoch in range(args.epoches):
        if epoch % 10 and epoch != args.epoches-1:
            continue
        start = time.time()
        images_dir = log_dir[args.num_disc] + "/Samples" + "/fid_buffer-epoch={}".format(epoch)
        images_path = pathlib.Path(images_dir)
        image_files = list(images_path.glob('*.jpg')) + list(images_path.glob('*.png'))

        imgs_list = []
        for image_file in image_files:
            img = imread(image_file)
            imgs_list.append(np.asarray(img))
        
        
        mean, std = get_inception_score(args, imgs_list)

        fid_score = calculate_fid_given_paths([images_dir, args.fid_stat], inception_path=args.inception_path)


        # summarywriter.add_scalar("IS_Score", mean, epoch+1)
        # summarywriter.add_scalar("FID_Score", fid_score, epoch+1)
        IS_Score_list.append(mean)
        FID_Score_list.append(fid_score)
        end = time.time()

        print("epoch: {}, is_score: {:.5f}, fid_score: {:.5f}, time: {:.5f}".format(epoch, mean, fid_score, end-start))

        content = "{:.5f}\t{:.5f}\n".format(fid_score, mean)
        log.write(content)
        log.flush()
"""
