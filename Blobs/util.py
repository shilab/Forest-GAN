import os
import tensorflow as tf
from model import BlobsGAN, BlobsWGAN
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import imageio
import glob


def create_dir(dir):
    try:
        if not os.path.exists(dir):
            os.makedirs(dir)
            print("creat a new dir: {}".format(dir))
        else:
            print("dir exists: {}".format(dir))
    except Exception as e:
        print(e)


def plot_and_save_dots(data_points, title, save_path, axis_scale=3):
    figure, axes = plt.subplots(nrows=1, ncols=1, figsize=(4.5, 4.5))
    axes.set_xlim(-axis_scale, axis_scale)
    axes.set_ylim(-axis_scale, axis_scale)
    axes.set_aspect('equal')
    axes.axis('on')
    axes.set_title(title)

    axes.scatter(
        data_points[:, 0],
        data_points[:, 1],
        marker='.',
        # c=cm.Set1(train_labels.astype(float) / 2.0 / 2.0),
        alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, transparent=True, bbox_inches='tight')

    plt.show()


def plot_and_save_imgs(imgs, title, save_path):
    figure = plt.figure(figsize=(15, 4))

    for i in range(16):
        plt.subplot(2, 8, i + 1)
        plt.imshow((imgs[i] + 1) / 2, cmap='gray')

        plt.axis("off")

    plt.suptitle(title, fontsize=25)

    if save_path is not None:
        plt.savefig(save_path, transparent=True, bbox_inches='tight')

    plt.show()


def plot_and_save_fig(dataset, title, save_path=None, axis_scale=3):
    # sample's dim
    if len(dataset[0].shape) == 1:
        # dots
        return plot_and_save_dots(dataset, title, save_path, axis_scale=axis_scale)
    else:
        # imgs
        return plot_and_save_imgs(dataset, title, save_path)


def generated_gif(fig_path, anim_file):
    fig_list = glob.glob(os.path.join(fig_path,'*'))
    fig_list.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0][5:]))

    with imageio.get_writer(anim_file, mode="I") as writer:
        for filename in fig_list:
            image = imageio.imread(filename)
            writer.append_data(image)
        image = imageio.imread(filename)
        writer.append_data(image)


def load_model(GAN, noise_dim, num_disc, lr, index):
    if GAN:
        blobs_model = BlobsGAN(noise_dim, num_disc, lr)
        generator = blobs_model.generator
        discriminators = blobs_model.discriminators
        storing_point = './training_checkpoints/GAN/num_disc_{}/check_points/epoch{}-{}'.format(num_disc, index*100, index)
    else:
        blobs_model = BlobsWGAN(noise_dim, num_disc, lr, grad_penalty_weight = 5)
        generator = blobs_model.generator
        discriminators = blobs_model.discriminators
        storing_point = './training_checkpoints/WGAN/num_disc_{}/check_points/epoch{}-{}'.format(num_disc, index*100, index)
        
    if num_disc == 1:
        checkpoint = tf.train.Checkpoint(generator=generator, discriminator0=discriminators[0])
    elif num_disc == 2:
        checkpoint = tf.train.Checkpoint(generator=generator, 
                                         discriminator0=discriminators[0],
                                         discriminator1=discriminators[1])
    elif num_disc == 5:
        checkpoint = tf.train.Checkpoint(generator=generator,
                                         discriminator0=discriminators[0],
                                         discriminator1=discriminators[1],
                                         discriminator2=discriminators[2],
                                         discriminator3=discriminators[3],
                                         discriminator4=discriminators[4])
    elif num_disc == 10:
        checkpoint = tf.train.Checkpoint(generator=generator,
                                         discriminator0=discriminators[0],
                                         discriminator1=discriminators[1],
                                         discriminator2=discriminators[2],
                                         discriminator3=discriminators[3],
                                         discriminator4=discriminators[4],
                                         discriminator5=discriminators[5],
                                         discriminator6=discriminators[6],
                                         discriminator7=discriminators[7],
                                         discriminator8=discriminators[8],
                                         discriminator9=discriminators[9])
    elif num_disc == 20:
        checkpoint = tf.train.Checkpoint(generator=generator,
                                         discriminator0=discriminators[0],
                                         discriminator1=discriminators[1],
                                         discriminator2=discriminators[2],
                                         discriminator3=discriminators[3],
                                         discriminator4=discriminators[4],
                                         discriminator5=discriminators[5],
                                         discriminator6=discriminators[6],
                                         discriminator7=discriminators[7],
                                         discriminator8=discriminators[8],
                                         discriminator9=discriminators[9],
                                         discriminator10=discriminators[10],
                                         discriminator11=discriminators[11],
                                         discriminator12=discriminators[12],
                                         discriminator13=discriminators[13],
                                         discriminator14=discriminators[14],
                                         discriminator15=discriminators[15],
                                         discriminator16=discriminators[16],
                                         discriminator17=discriminators[17],
                                         discriminator18=discriminators[18],
                                         discriminator19=discriminators[19])
    elif num_disc == 50:
        checkpoint = tf.train.Checkpoint(generator=generator,
                                         discriminator0=discriminators[0],
                                         discriminator1=discriminators[1],
                                         discriminator2=discriminators[2],
                                         discriminator3=discriminators[3],
                                         discriminator4=discriminators[4],
                                         discriminator5=discriminators[5],
                                         discriminator6=discriminators[6],
                                         discriminator7=discriminators[7],
                                         discriminator8=discriminators[8],
                                         discriminator9=discriminators[9],
                                         discriminator10=discriminators[10],
                                         discriminator11=discriminators[11],
                                         discriminator12=discriminators[12],
                                         discriminator13=discriminators[13],
                                         discriminator14=discriminators[14],
                                         discriminator15=discriminators[15],
                                         discriminator16=discriminators[16],
                                         discriminator17=discriminators[17],
                                         discriminator18=discriminators[18],
                                         discriminator19=discriminators[19],
                                         discriminator20=discriminators[20],
                                         discriminator21=discriminators[21],
                                         discriminator22=discriminators[22],
                                         discriminator23=discriminators[23],
                                         discriminator24=discriminators[22],
                                         discriminator25=discriminators[25],
                                         discriminator26=discriminators[26],
                                         discriminator27=discriminators[27],
                                         discriminator28=discriminators[28],
                                         discriminator29=discriminators[29],
                                         discriminator30=discriminators[30],
                                         discriminator31=discriminators[31],
                                         discriminator32=discriminators[32],
                                         discriminator33=discriminators[33],
                                         discriminator34=discriminators[34],
                                         discriminator35=discriminators[35],
                                         discriminator36=discriminators[36],
                                         discriminator37=discriminators[37],
                                         discriminator38=discriminators[38],
                                         discriminator39=discriminators[39],
                                         discriminator40=discriminators[40],
                                         discriminator41=discriminators[41],
                                         discriminator42=discriminators[42],
                                         discriminator43=discriminators[43],
                                         discriminator44=discriminators[44],
                                         discriminator45=discriminators[45],
                                         discriminator46=discriminators[46],
                                         discriminator47=discriminators[47],
                                         discriminator48=discriminators[48],
                                         discriminator49=discriminators[49])

    checkpoint.restore(storing_point)
    
    return blobs_model
