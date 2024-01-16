import tensorflow as tf
from tensorflow.keras import datasets, layers, Sequential, Model
from tensorflow.keras.losses import BinaryCrossentropy


class CirclesGAN(object):
    """docstring for CirclesGAN"""
    def __init__(self, noise_dim, num_disc, lr, decay=False):
        # Hyper-parameters
        self.noise_dim = noise_dim
        self.num_disc = num_disc

        # model
        self.generator = self.make_generator_model(noise_dim)
        self.discriminators = self.make_multi_disc_model(num_disc)

        # optimizer
        self.generator_optimizer, self.discriminator_optimizer = self.optimizer(lr, decay)

        # cross_entropy_loss
        self.cross_entropy = BinaryCrossentropy(from_logits=True)

    def make_generator_model(self, noise_dim):
        model = tf.keras.Sequential()
        model.add(layers.Dense(256, input_shape=(noise_dim, )))
        model.add(layers.LeakyReLU())

        model.add(layers.Dense(256))
        model.add(layers.LeakyReLU())

        model.add(layers.Dense(2))
        assert model.output_shape == (None, 2)  # Note: None is the batch size

        return model

    def make_discriminator_model(self):
        model = tf.keras.Sequential()
        model.add(layers.Dense(256, input_shape=(2, )))
        model.add(layers.LeakyReLU())

        model.add(layers.Dense(256))
        model.add(layers.LeakyReLU())

        model.add(layers.Dense(1, activation='sigmoid'))

        return model

    def make_multi_disc_model(self, num_disc):
        discriminators = []
        for i in range(num_disc):
            discriminators.append(self.make_discriminator_model())

        return discriminators

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss

        return total_loss

    def generator_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)

    def optimizer(self, lr, decay):        
        if decay:
            # Learning Rate for DISCRIMINATOR
            LR_DISC = tf.compat.v1.train.polynomial_decay(learning_rate=1e-2,
                                                          global_step=tf.compat.v1.train.get_or_create_global_step(),
                                                          decay_steps=10000,
                                                          end_learning_rate=1e-5,
                                                          power=0.5,
                                                          cycle=True)
            generator_optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.5, beta_2=0.999)
            discriminator_optimizer = tf.keras.optimizers.Adam(LR_DISC, beta_1=0.5, beta_2=0.999)
        else:
            # Adam
            # generator_optimizer = tf.keras.optimizers.Adam(1e-4)
            # discriminator_optimizer=tf.keras.optimizers.Adam(1e-4)
            generator_optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.5, beta_2=0.999)
            discriminator_optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.5, beta_2=0.999)

        # RMSprop
        # generator_optimizer = tf.keras.optimizers.RMSprop(lr)
        # discriminator_optimizer = tf.keras.optimizers.RMSprop(lr)

        return generator_optimizer, discriminator_optimizer


class CirclesWGAN(object):
    """docstring for CirclesWGAN"""
    def __init__(self, noise_dim, num_disc, lr, grad_penalty_weight, decay=False):
        # Hyper-parameters
        self.noise_dim = noise_dim
        self.num_disc = num_disc
        self.grad_penalty_weight = grad_penalty_weight

        # model
        self.generator = self.make_generator_model(noise_dim)
        self.discriminators = self.make_multi_disc_model(num_disc)

        # optimizer
        self.generator_optimizer, self.discriminator_optimizer = self.optimizer(lr, decay)

    def make_generator_model(self, noise_dim):
        model = tf.keras.Sequential()
        model.add(layers.Dense(256, input_shape=(noise_dim, )))
        model.add(layers.ReLU())

        model.add(layers.Dense(256))
        model.add(layers.ReLU())

        model.add(layers.Dense(2))

        return model

    def make_discriminator_model(self):
        model = tf.keras.Sequential()
        model.add(layers.Dense(256, input_shape=(2, )))
        model.add(layers.ReLU())

        model.add(layers.Dense(256))
        model.add(layers.ReLU())

        model.add(layers.Dense(1))

        return model

    def make_multi_disc_model(self, num_disc):
        discriminators = []
        for i in range(num_disc):
            discriminators.append(self.make_discriminator_model())

        return discriminators

    def generator_loss(self, fake_output):
        return -tf.reduce_mean(fake_output)

    def discriminator_loss(self, real_output, fake_output):
        return -tf.reduce_mean(real_output) + tf.reduce_mean(fake_output)

    def optimizer(self, lr, decay):
        # if decay:
        #     # Learning Rate for DISCRIMINATOR
        #     LR_DISC = tf.compat.v1.train.polynomial_decay(learning_rate=1e-2,
        #                                                   global_step=tf.compat.v1.train.get_or_create_global_step(),
        #                                                   decay_steps=10000,
        #                                                   end_learning_rate=1e-5,
        #                                                   power=0.5,
        #                                                   cycle=True)
        #     generator_optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.5, beta_2=0.999)
        #     discriminator_optimizer = tf.keras.optimizers.Adam(LR_DISC, beta_1=0.5, beta_2=0.999)
        # else:
        #     # Adam
        #     # generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        #     # discriminator_optimizer=tf.keras.optimizers.Adam(1e-4)
        #     generator_optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.5, beta_2=0.999)
        #     discriminator_optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.5, beta_2=0.999)

        # RMSprop
        generator_optimizer = tf.keras.optimizers.RMSprop(lr)
        discriminator_optimizer = tf.keras.optimizers.RMSprop(lr)

        return generator_optimizer, discriminator_optimizer


class BlobsGAN(object):
    """docstring for BlobsGAN"""
    def __init__(self, noise_dim, num_disc, lr, decay=False, layers=[1,1,1,1]):
        # Hyper-parameters
        self.noise_dim = noise_dim
        self.num_disc = num_disc

        self.layer1_gen = layers[0]
        self.layer2_gen = layers[1]

        self.layer1_disc = layers[2]
        self.layer2_disc = layers[3]
    
        # model
        self.generator = self.make_generator_model(noise_dim)
        self.discriminators = self.make_multi_disc_model(num_disc)

        # optimizer
        self.generator_optimizer, self.discriminator_optimizer = self.optimizer(lr, decay)

        # cross_entropy_loss
        self.cross_entropy = BinaryCrossentropy(from_logits=True)

    def make_generator_model(self, noise_dim):
        model = tf.keras.Sequential()
        model.add(layers.Dense(int(256*self.layer1_gen), input_shape=(noise_dim, )))
        model.add(layers.LeakyReLU())

        model.add(layers.Dense(int(256*self.layer2_gen)))
        model.add(layers.LeakyReLU())

        model.add(layers.Dense(2))
        assert model.output_shape == (None, 2)  # Note: None is the batch size

        return model

    def make_discriminator_model(self):
        model = tf.keras.Sequential()
        model.add(layers.Dense(int(256*self.layer1_disc), input_shape=(2, )))
        model.add(layers.LeakyReLU())

        model.add(layers.Dense(int(256*self.layer2_disc)))
        model.add(layers.LeakyReLU())

        model.add(layers.Dense(1, activation='sigmoid'))

        return model

    def make_multi_disc_model(self, num_disc):
        discriminators = []
        for i in range(num_disc):
            discriminators.append(self.make_discriminator_model())

        return discriminators

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss

        return total_loss

    def generator_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)

    def optimizer(self, lr, decay):
        # RMSprop
        # generator_optimizer = tf.keras.optimizers.RMSprop(lr)
        # discriminator_optimizer = tf.keras.optimizers.RMSprop(lr)

        if decay:
            # Learning Rate for DISCRIMINATOR
            LR_DISC = tf.compat.v1.train.polynomial_decay(learning_rate=1e-2,
                                                          global_step=tf.compat.v1.train.get_or_create_global_step(),
                                                          decay_steps=10000,
                                                          end_learning_rate=1e-5,
                                                          power=0.5,
                                                          cycle=True)
            generator_optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.5, beta_2=0.999)
            
            discriminator_optimizer = []
            for i in range(self.num_disc):
                dis_optimizer = tf.keras.optimizers.Adam(LR_DISC, beta_1=0.5, beta_2=0.999)
                discriminator_optimizer.append(dis_optimizer)
                
        else:
            # Adam
            # generator_optimizer = tf.keras.optimizers.Adam(1e-4)
            # discriminator_optimizer=tf.keras.optimizers.Adam(1e-4)
            generator_optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.5, beta_2=0.999)
            
            discriminator_optimizer = []
            for i in range(self.num_disc):
                dis_optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.5, beta_2=0.999)
                discriminator_optimizer.append(dis_optimizer)
                
        return generator_optimizer, discriminator_optimizer


class BlobsWGAN(object):
    """docstring for BlobsWGAN"""
    def __init__(self, noise_dim, num_disc, lr, grad_penalty_weight, decay=False, layers=[1,1,1,1]):
        # Hyper-parameters
        self.noise_dim = noise_dim
        self.num_disc = num_disc
        
        self.layer1_gen = layers[0]
        self.layer2_gen = layers[1]

        self.layer1_disc = layers[2]
        self.layer2_disc = layers[3]
            
        self.grad_penalty_weight = grad_penalty_weight

        # model
        self.generator = self.make_generator_model(noise_dim)
        self.discriminators = self.make_multi_disc_model(num_disc)

        # optimizer
        self.generator_optimizer, self.discriminator_optimizer = self.optimizer(lr, decay)

    def make_generator_model(self, noise_dim):
        model = tf.keras.Sequential()
        model.add(layers.Dense(int(256*self.layer1_gen), input_shape=(noise_dim, )))
        model.add(layers.ReLU())

        model.add(layers.Dense(int(256*self.layer2_gen)))
        model.add(layers.ReLU())

        model.add(layers.Dense(2))

        return model

    def make_discriminator_model(self):
        model = tf.keras.Sequential()
        model.add(layers.Dense(int(256*self.layer1_disc), input_shape=(2, )))
        model.add(layers.ReLU())

        model.add(layers.Dense(int(256*self.layer2_disc)))
        model.add(layers.ReLU())

        model.add(layers.Dense(1))
        
        return model

    def make_multi_disc_model(self, num_disc):
        discriminators = []
        for i in range(num_disc):
            discriminators.append(self.make_discriminator_model())

        return discriminators

    def generator_loss(self, fake_output):
        return -tf.reduce_mean(fake_output)

    def discriminator_loss(self, real_output, fake_output):
        return -tf.reduce_mean(real_output) + tf.reduce_mean(fake_output)

    def optimizer(self, lr, decay):
        # if decay:
        #     # Learning Rate for DISCRIMINATOR
        #     LR_DISC = tf.compat.v1.train.polynomial_decay(learning_rate=1e-2,
        #                                                   global_step=tf.compat.v1.train.get_or_create_global_step(),
        #                                                   decay_steps=10000,
        #                                                   end_learning_rate=1e-5,
        #                                                   power=0.5,
        #                                                   cycle=True)
        #     generator_optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.5, beta_2=0.999)
        #     discriminator_optimizer = tf.keras.optimizers.Adam(LR_DISC, beta_1=0.5, beta_2=0.999)
        # else:
        #     # Adam
        #     # generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        #     # discriminator_optimizer=tf.keras.optimizers.Adam(1e-4)
        #     generator_optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.5, beta_2=0.999)
        #     discriminator_optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.5, beta_2=0.999)

        # RMSprop
        generator_optimizer = tf.keras.optimizers.RMSprop(lr)
        discriminator_optimizer = []
        for i in range(self.num_disc):
            dis_optimizer = tf.keras.optimizers.RMSprop(lr)
            discriminator_optimizer.append(dis_optimizer)
            
        return generator_optimizer, discriminator_optimizer


class MNISTDCGAN(object):
    """docstring for MNISTDCGAN"""
    def __init__(self, noise_dim, num_disc, lr, decay=False):
        # Hyper-parameters
        self.noise_dim = noise_dim
        self.num_disc = num_disc

        # model
        self.generator = self.make_generator_model(noise_dim)
        self.discriminators = self.make_multi_disc_model(num_disc)

        # optimizer
        self.generator_optimizer, self.discriminator_optimizer = self.optimizer(lr, decay)

        # cross_entropy_loss
        self.cross_entropy = BinaryCrossentropy(from_logits=True)

    def make_generator_model(self, noise_dim):
        model = tf.keras.Sequential()
        model.add(layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(noise_dim, )))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Reshape((7, 7, 256)))
        assert model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size

        model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding="same"))
        assert model.output_shape == (None, 7, 7, 128)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding="same"))
        assert model.output_shape == (None, 14, 14, 64)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding="same", activation="tanh"))
        assert model.output_shape == (None, 28, 28, 1)

        return model

    def make_discriminator_model(self):
        model = tf.keras.Sequential()
        model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Flatten())
        # model.add(layers.Dropout(0.3))
        model.add(layers.Dense(1, activation='sigmoid'))

        return model

    def make_multi_disc_model(self, num_disc):
        discriminators = []
        for i in range(num_disc):
            discriminators.append(self.make_discriminator_model())

        return discriminators

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss

        return total_loss

    def generator_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)

    def optimizer(self, lr, decay):
        # RMSprop
        # generator_optimizer = tf.keras.optimizers.RMSprop(lr)
        # discriminator_optimizer = tf.keras.optimizers.RMSprop(lr)

        if decay:
            # Learning Rate for DISCRIMINATOR
            LR_DISC = tf.compat.v1.train.polynomial_decay(learning_rate=1e-2,
                                                          global_step=tf.compat.v1.train.get_or_create_global_step(),
                                                          decay_steps=10000,
                                                          end_learning_rate=1e-5,
                                                          power=0.5,
                                                          cycle=True)
            generator_optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.5, beta_2=0.999)
            discriminator_optimizer = tf.keras.optimizers.Adam(LR_DISC, beta_1=0.5, beta_2=0.999)
        else:
            # Adam
            # generator_optimizer = tf.keras.optimizers.Adam(1e-4)
            # discriminator_optimizer=tf.keras.optimizers.Adam(1e-4)
            generator_optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.5, beta_2=0.999)
            discriminator_optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.5, beta_2=0.999)

        return generator_optimizer, discriminator_optimizer


class MNISTWGAN(object):
    """docstring for BlobsWGAN"""
    def __init__(self, noise_dim, num_disc, lr, grad_penalty_weight, decay=False):
        # Hyper-parameters
        self.noise_dim = noise_dim
        self.num_disc = num_disc
        self.grad_penalty_weight = grad_penalty_weight

        # model
        self.generator = self.make_generator_model(noise_dim)
        self.discriminators = self.make_multi_disc_model(num_disc)

        # optimizer
        self.generator_optimizer, self.discriminator_optimizer = self.optimizer(lr, decay)

    def make_generator_model(self, noise_dim):
        model = tf.keras.Sequential()
        model.add(layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(100, )))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Reshape((7, 7, 256)))
        assert model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size

        model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same'))
        assert model.output_shape == (None, 7, 7, 128)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same'))
        assert model.output_shape == (None, 14, 14, 64)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', activation='tanh'))
        assert model.output_shape == (None, 28, 28, 1)

        return model

    def make_discriminator_model(self):
        model = tf.keras.Sequential()
        model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Flatten())
        model.add(layers.Dense(1))

        return model

    def make_multi_disc_model(self, num_disc):
        discriminators = []
        for i in range(num_disc):
            discriminators.append(self.make_discriminator_model())

        return discriminators

    def generator_loss(self, fake_output):
        return -tf.reduce_mean(fake_output)

    def discriminator_loss(self, real_output, fake_output):
        return -tf.reduce_mean(real_output) + tf.reduce_mean(fake_output)

    def optimizer(self, lr, decay):
        if decay:
            # Learning Rate for DISCRIMINATOR
            LR_DISC = tf.compat.v1.train.polynomial_decay(learning_rate=1e-2,
                                                          global_step=tf.compat.v1.train.get_or_create_global_step(),
                                                          decay_steps=10000,
                                                          end_learning_rate=1e-5,
                                                          power=0.5,
                                                          cycle=True)
            generator_optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.5, beta_2=0.999)
            discriminator_optimizer = tf.keras.optimizers.Adam(LR_DISC, beta_1=0.5, beta_2=0.999)
        else:
            # Adam
            # generator_optimizer = tf.keras.optimizers.Adam(1e-4)
            # discriminator_optimizer=tf.keras.optimizers.Adam(1e-4)
            generator_optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.5, beta_2=0.999)
            discriminator_optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.5, beta_2=0.999)

        # RMSprop
        generator_optimizer = tf.keras.optimizers.RMSprop(1e-5)
        discriminator_optimizer = tf.keras.optimizers.RMSprop(1e-5)

        return generator_optimizer, discriminator_optimizer


class CIFAR10DCGAN(object):
    """docstring for BlobsDCGAN"""
    def __init__(self, noise_dim, num_disc, lr, decay=False):
        # Hyper-parameters
        self.noise_dim = noise_dim
        self.num_disc = num_disc

        # model
        self.generator = self.make_generator_model(noise_dim)
        self.discriminators = self.make_multi_disc_model(num_disc)

        # optimizer
        self.generator_optimizer, self.discriminator_optimizer = self.optimizer(lr, decay)

        # cross_entropy_loss
        self.cross_entropy = BinaryCrossentropy(from_logits=True)

    def make_generator_model(self, noise_dim):
        model = tf.keras.Sequential()
        model.add(layers.Dense(4 * 4 * 256, use_bias=False, input_shape=(noise_dim, )))

        model.add(layers.Reshape((4, 4, 256)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(256, (5, 5), strides=(1, 1), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', activation='tanh'))
        assert model.output_shape == (None, 32, 32, 3)

        return model

    def make_discriminator_model(self):
        model = tf.keras.Sequential()
        model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[32, 32, 3]))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        # model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        # model.add(layers.Dropout(0.3))

        model.add(layers.Flatten())
        model.add(layers.Dense(1, activation='sigmoid'))

        return model

    def make_multi_disc_model(self, num_disc):
        discriminators = []
        for i in range(num_disc):
            discriminators.append(self.make_discriminator_model())

        return discriminators

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss

        return total_loss

    def generator_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)

    def optimizer(self, lr, decay):
        if decay:
            # Learning Rate for DISCRIMINATOR
            LR_DISC = tf.compat.v1.train.polynomial_decay(learning_rate=1e-2,
                                                          global_step=tf.compat.v1.train.get_or_create_global_step(),
                                                          decay_steps=10000,
                                                          end_learning_rate=1e-5,
                                                          power=0.5,
                                                          cycle=True)
            generator_optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.5, beta_2=0.999)
            discriminator_optimizer = tf.keras.optimizers.Adam(LR_DISC, beta_1=0.5, beta_2=0.999)
        else:
            # Adam
            # generator_optimizer = tf.keras.optimizers.Adam(1e-4)
            # discriminator_optimizer=tf.keras.optimizers.Adam(1e-4)
            generator_optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.5, beta_2=0.999)
            discriminator_optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.5, beta_2=0.999)

        # RMSprop
        # generator_optimizer = tf.keras.optimizers.RMSprop(lr)
        # discriminator_optimizer = tf.keras.optimizers.RMSprop(lr)

        return generator_optimizer, discriminator_optimizer

class CIFAR10WGAN(object):
    """docstring for BlobsWGAN"""
    def __init__(self, noise_dim, num_disc, lr, grad_penalty_weight, decay=False):
        # Hyper-parameters
        self.noise_dim = noise_dim
        self.num_disc = num_disc
        self.grad_penalty_weight = grad_penalty_weight

        # model
        self.generator = self.make_generator_model(noise_dim)
        self.discriminators = self.make_multi_disc_model(num_disc)

        # optimizer
        self.generator_optimizer, self.discriminator_optimizer = self.optimizer(lr, decay)

    def make_generator_model(self, noise_dim):
        model = tf.keras.Sequential()
        model.add(layers.Dense(4 * 4 * 256, use_bias=False, input_shape=(noise_dim, )))

        model.add(layers.Reshape((4, 4, 256)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(256, (5, 5), strides=(1, 1), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', activation='tanh'))
        assert model.output_shape == (None, 32, 32, 3)

        return model

    def make_discriminator_model(self):
        model = tf.keras.Sequential()
        model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[32, 32, 3]))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        # model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        # model.add(layers.Dropout(0.3))

        model.add(layers.Flatten())
        model.add(layers.Dense(1))

        return model

    def make_multi_disc_model(self, num_disc):
        discriminators = []
        for i in range(num_disc):
            discriminators.append(self.make_discriminator_model())

        return discriminators

    def generator_loss(self, fake_output):
        return -tf.reduce_mean(fake_output)

    def discriminator_loss(self, real_output, fake_output):
        return -tf.reduce_mean(real_output) + tf.reduce_mean(fake_output)

    def optimizer(self, lr, decay):
        if decay:
            # Learning Rate for DISCRIMINATOR
            LR_DISC = tf.compat.v1.train.polynomial_decay(learning_rate=1e-2,
                                                          global_step=tf.compat.v1.train.get_or_create_global_step(),
                                                          decay_steps=10000,
                                                          end_learning_rate=1e-5,
                                                          power=0.5,
                                                          cycle=True)
            generator_optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.5, beta_2=0.999)
            discriminator_optimizer = tf.keras.optimizers.Adam(LR_DISC, beta_1=0.5, beta_2=0.999)
        else:
            # Adam
            # generator_optimizer = tf.keras.optimizers.Adam(1e-4)
            # discriminator_optimizer=tf.keras.optimizers.Adam(1e-4)
            generator_optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.5, beta_2=0.999)
            discriminator_optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.5, beta_2=0.999)

        # RMSprop
        generator_optimizer = tf.keras.optimizers.RMSprop(1e-5)
        discriminator_optimizer = tf.keras.optimizers.RMSprop(1e-5)

        return generator_optimizer, discriminator_optimizer
