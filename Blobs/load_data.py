import pickle
import numpy as np
import tensorflow as tf


def load_circles(train_size):
    with open('noisy_circles_data.pkl', 'rb') as f:
        noisy_circles_dataset = pickle.load(f)

    train_dataset = noisy_circles_dataset[0][:train_size].astype('float32')
    train_labels = noisy_circles_dataset[1][:train_size].astype('float32')

    test_dataset = noisy_circles_dataset[0][train_size:].astype('float32')
    test_labels = noisy_circles_dataset[1][train_size:].astype('float32')

    return (train_dataset, train_labels), (test_dataset, test_labels)


def load_blobs(path, train_size):
    with open(path, 'rb') as f:
        gmm_blob_data = pickle.load(f)

    train_dataset = gmm_blob_data['samples'][:train_size].astype('float32')
    train_labels = gmm_blob_data['labels'][:train_size].astype('float32')
    train_density = gmm_blob_data['density'][:train_size].astype('float32')

    test_dataset = gmm_blob_data['samples'][train_size:].astype('float32')
    test_labels = gmm_blob_data['labels'][train_size:].astype('float32')
    test_density = gmm_blob_data['density'][train_size:].astype('float32')

    return (train_dataset, train_labels, train_density), (test_dataset, test_labels, test_density)


def load_MNIST():
    (train_MNIST, train_labels), (test_MNIST, test_labels) = tf.keras.datasets.mnist.load_data()
    train_dataset = train_MNIST.reshape(train_MNIST.shape[0], 28, 28, 1).astype("float32")
    train_dataset = (train_dataset - 127.5) / 127.5  # Normalize the images to [-1, 1]
    test_dataset = test_MNIST.reshape(test_MNIST.shape[0], 28, 28, 1).astype("float32")
    test_dataset = (test_dataset - 127.5) / 127.5  # Normalize the images to [-1, 1]

    return (train_dataset, train_labels), (test_dataset, test_labels)

def load_CIFAR10():
    (train_CIFAR10, train_labels), (test_CIFAR10, test_labels) = tf.keras.datasets.cifar10.load_data()
    train_dataset = train_CIFAR10.astype('float32')
    train_dataset = (train_dataset - 127.5) / 127.5  # Normalize the images to [-1, 1]
    test_dataset = test_CIFAR10.astype('float32')
    test_dataset = (test_dataset - 127.5) / 127.5  # Normalize the images to [-1, 1]

    return (train_dataset, train_labels), (test_dataset, test_labels)

def load_CIFAR100():
    (train_CIFAR100, train_labels), (test_CIFAR100, test_labels) = tf.keras.datasets.cifar100.load_data()
    train_dataset = train_CIFAR100.astype('float32')
    train_dataset = (train_dataset - 127.5) / 127.5  # Normalize the images to [-1, 1]
    test_dataset = test_CIFAR100.astype('float32')
    test_dataset = (test_dataset - 127.5) / 127.5  # Normalize the images to [-1, 1]

    return (train_dataset, train_labels), (test_dataset, test_labels)



def get_bootstrap_bags(dataset, n_bags, random_state):
    """Generate bootstrap bags using the bootstrap method."""
    r = np.random.RandomState(random_state)
    indices = r.randint(0, len(dataset), (n_bags, len(dataset)))
    bootstrap_bags = dataset[indices]

    return indices, bootstrap_bags
