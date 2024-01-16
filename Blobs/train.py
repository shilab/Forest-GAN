import os
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import time
from IPython import display
import functools
from util import plot_and_save_fig


def gradient_penalty(f, real, fake):
    alpha = tf.random.uniform([real.shape[0], 1], 0., 1.)
    diff = fake - real
    inter = real + (alpha * diff)
    with tf.GradientTape() as t:
        t.watch(inter)
        pred = f(inter)
    grad = t.gradient(pred, [inter])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1]))
    gp = tf.reduce_mean((slopes - 1.)**2)
    return gp


def train(model,
          bootstrap_bags,
          epochs,
          batch_size,
          fig_dir=None,
          checkpoint=None,
          checkpoint_dir=None,
          iteration_D=2,
          penalty=False, 
          loss_log=""):
    

    loss_log = open(loss_log, "w")
    content = "gen_loss\t"
    for i in range(model.num_disc):
        content += "disc-{}_loss\t".format(i)
    content += "\n"
    loss_log.write(content)
    loss_log.flush()

    # We will reuse this seed overtime (so it's easier)
    # to visualize progress in the animated GIF)
    seed = tf.random.normal([10000, model.noise_dim])

    # Batch and shuffle all datasets
    train_batches = []
    for i in range(model.num_disc):
        train_batches.append(
            tf.data.Dataset.from_tensor_slices(bootstrap_bags[i]).shuffle(bootstrap_bags[i].shape[0]).batch(
                batch_size, drop_remainder=True))

    gen_loss_log, disc_loss_log = [], []
    for epoch in range(epochs):
        count = 0
        gen_loss_sum = 0.0
        disc_loss_sum = [0.0 for _ in range(model.num_disc)]
        start = time.time()
        for zipped_batches in zip(*train_batches):
            gen_loss, all_disc_loss = train_step(model, batch_size, zipped_batches, iteration_D, penalty)
            
            count += 1
            gen_loss_sum += gen_loss.numpy()
            for i in range(model.num_disc):
                disc_loss_sum[i] += all_disc_loss[i].numpy()
        end = time.time()
        gen_loss = np.array(gen_loss_sum) / count
        disc_loss = np.array(disc_loss_sum) / count

        if (epoch + 1) % 10 == 0:
            # Produce images for the GIF
            display.clear_output(wait=True)
            generated_points = model.generator(seed, training=False)
            plot_and_save_fig(generated_points, 'epoch {}'.format(epoch + 1),
                              os.path.join(fig_dir, 'epoch{}.png'.format(epoch + 1)))

        # Save the model
        if checkpoint is not None:
            if (epoch + 1) % 100 == 0:
                checkpoint.save(file_prefix=os.path.join(checkpoint_dir, 'epoch{}'.format(epoch + 1)))

        content = "{:.5f}\t".format(gen_loss)
        for i in range(model.num_disc):
            content += "{:.5f}\t".format(disc_loss[i])
        
        loss_log.write(content+"\n")
        loss_log.flush()

        print("epoch-{}; {}; tmes:{:.4f}s".format(epoch, content, end-start), end="\n")

    return gen_loss_log, disc_loss_log


# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(model, batch_size, zipped_real_samples, iteration_D, penalty):
    # update D
    for _ in range(iteration_D):
        all_disc_loss = []
        # gp = []
        # for each discriminator
        for i in range(model.num_disc):
            noise = tf.random.normal([batch_size, model.noise_dim])
            with tf.GradientTape() as disc_tape:
                generated_samples = model.generator(noise, training=True)
                real_output = model.discriminators[i](zipped_real_samples[i], training=True)
                fake_output = model.discriminators[i](generated_samples, training=True)

                disc_loss = model.discriminator_loss(real_output, fake_output)
                # gradient penalty for WGAN-GP
                if penalty == True:
                    gp_temp = gradient_penalty(functools.partial(model.discriminators[i], training=True), zipped_real_samples[i],
                                          generated_samples)
                    # gp.append(gp_temp)
                    # if gp[i].numpy() != gp[i].numpy():
                    #     count=1 

                    disc_loss += model.grad_penalty_weight * gp_temp

                gradients_of_discriminator = disc_tape.gradient(disc_loss, model.discriminators[i].trainable_variables)
                model.discriminator_optimizer[i].apply_gradients(zip(gradients_of_discriminator, model.discriminators[i].trainable_variables))
            all_disc_loss.append(disc_loss)

    # update G
    for _ in range(1):
        with tf.GradientTape() as gen_tape:
            all_gen_loss = []
            for i in range(model.num_disc):
                noise = tf.random.normal([batch_size, model.noise_dim])
                generated_samples = model.generator(noise, training=True)
                fake_output = model.discriminators[i](generated_samples, training=True)
                gen_loss = model.generator_loss(fake_output)

                all_gen_loss.append(gen_loss)

            avg_gen_loss = tf.reduce_mean(all_gen_loss)
            gradients_of_generator = gen_tape.gradient(avg_gen_loss, model.generator.trainable_variables)
            model.generator_optimizer.apply_gradients(zip(gradients_of_generator, model.generator.trainable_variables))

    # if avg_gen_loss != avg_gen_loss:
    #     count = 1
    #     print(avg_gen_loss)

    # if avg_gen_loss > 0.0:
    #     count  = 1
    return avg_gen_loss, all_disc_loss