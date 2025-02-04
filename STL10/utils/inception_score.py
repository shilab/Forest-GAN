# Code derived from tensorflow/tensorflow/models/image/imagenet/classify_image.py
from __future__ import absolute_import, division, print_function

import math
import os
import os.path
import sys
import tarfile

import numpy as np
# import tensorflow as tf           # temsorflow2.0

import tensorflow.compat.v1 as tf   # temsorflow2.0

from six.moves import urllib
from tqdm import tqdm

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
MODEL_DIR = "./Inception_Net/is_score/"

softmax = None

tf.disable_eager_execution()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# config = tf.ConfigProto(log_device_placement=True)
# config.gpu_options.visible_device_list= '0'
# config.gpu_options.allow_growth = True

# Call this function with list of images. Each of elements should be a
# numpy array with values ranging from 0 to 255.

def get_inception_score(args, images, splits=10):
    # _init_inception(args)
    assert type(images) == list
    assert type(images[0]) == np.ndarray
    assert len(images[0].shape) == 3
    assert np.max(images[0]) > 10
    assert np.min(images[0]) >= 0.0
    inps = []
    for img in images:
        img = img.astype(np.float32)
        inps.append(np.expand_dims(img, 0))
    bs = 256
    with tf.Session(config=config) as sess:
    # with tf.InteractiveSession(config=config) as sess:
        preds = []
        n_batches = int(math.ceil(float(len(inps)) / float(bs)))
        for i in range(n_batches):
            sys.stdout.flush()
            inp = inps[(i * bs) : min((i + 1) * bs, len(inps))]
            inp = np.concatenate(inp, 0)
            pred = sess.run(softmax, {"ExpandDims:0": inp}) 
            preds.append(pred)
        preds = np.concatenate(preds, 0)
        scores = []
        for i in range(splits): 
            part = preds[(i * preds.shape[0] // splits) : ((i + 1) * preds.shape[0] // splits), :]
            kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
            kl = np.mean(np.sum(kl, 1))
            scores.append(np.exp(kl))

        sess.close()
    return np.mean(scores), np.std(scores)


# This function is called automatically.
def _init_inception(args):
    global softmax
        
    filepath = args.inception_path
    model_dir = os.path.join(filepath, "is_score")
    if not os.path.exists(os.path.join(model_dir, "classify_image_graph_def.pb")):

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        inception_net = os.path.join(filepath, "inception-2015-12-05.tgz")
        tarfile.open(inception_net, "r:gz").extractall(model_dir)

    with tf.gfile.FastGFile(os.path.join(model_dir, "classify_image_graph_def.pb"), "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name="")
    # Works with an arbitrary minibatch size.

    with tf.Session(config=config) as sess:
    # with tf.InteractiveSession(config=config) as sess:
        pool3 = sess.graph.get_tensor_by_name("pool_3:0")
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

        softmax = tf.nn.softmax(logits)

