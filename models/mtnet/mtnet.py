import os
import sys
import time
import numpy as np
import tensorflow as tf
from bunch import Bunch
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Activation, BatchNormalization, Flatten, Dense, Add, UpSampling2D
from tensorflow.keras import Input, Model, Sequential

import utils.config_val as g_config
from models.wrapper.layer_caffe import *


class Mtnet():
    def __init__(self):
        self.conv_count = -1

    def update_count(self):
        self.conv_count += 1

    def clear_count(self):
        self.conv_count = -1

    def conv2d(self, input, filters, strides, kernel_size=5, padding="same", activation="relu", prefix="", name=None, use_bias=True):
        if name is None:
            self.update_count()
            count = "" if 0 == self.conv_count else "_" + str(self.conv_count)
            name = prefix + "conv" + count

        x = Conv2D(filters=filters, strides=strides, kernel_size=kernel_size, padding=padding, activation=activation, name=name, use_bias=use_bias)(input)
        return x

    def add(self, input, prefix="", name=None):
        if name is None:
            name = prefix + "eltwise"
        x = Add(name=name)(input)
        return x

    def resnet(self, input, filters_1, filters_2, strides, is_skip=True, prefix=""):
        x1 = self.conv2d(input, filters=filters_1, strides=strides, prefix=prefix)
        x1 = self.conv2d(x1,filters=filters_2, strides=1, prefix=prefix)
        if is_skip:
            x2 = input
        else:
            x2 = self.conv2d(input, filters=filters_2, strides=strides, prefix=prefix)
        return self.add([x1, x2], prefix=prefix)

    def upconv(self, input, filters, strides, prefix="", name=None):
        x = UpSampling2D(size=(2, 2))(input)
        x = self.conv2d(x, filters=filters, strides=strides, activation=None, prefix=prefix, name=name)
        return x

    def gen(self, input_shape):
        # create inputs
        input = tf.keras.Input(shape=(input_shape[1], input_shape[2], input_shape[3]))
        # linear layer should be replace with scale in caffe
        x = Activation("linear")(input)
        # create layers
        self.clear_count()
        x = self.conv2d(x, filters=8, strides=2, prefix="s1_")
        self.clear_count()
        x = self.conv2d(x, filters=8, strides=2, prefix="s2_")
        self.clear_count()
        x = self.conv2d(x, filters=24, strides=2, prefix="s3_")

        self.clear_count()
        x = self.resnet(x, 24, 48, strides=2, is_skip=False, prefix="s4_b1_")
        self.clear_count()
        x = self.resnet(x, 32, 56, strides=2, is_skip=False, prefix="s5_b1_")
        self.clear_count()
        x = self.resnet(x, 32, 104, strides=1, is_skip=False, prefix="s6_b1_")

        self.clear_count()
        x = self.resnet(x, 32, 104, strides=1, is_skip=True, prefix="s6_b2_")
        self.clear_count()
        x = self.resnet(x, 32, 104, strides=1, is_skip=True, prefix="s6_b3_")
        self.clear_count()
        i1 = self.resnet(x, 32, 104, strides=1, is_skip=True, prefix="s6_b4_")


        # stage-7 b-1
        self.clear_count()
        x1 = self.resnet(i1, 32, 104, strides=2, is_skip=False, prefix="s7_b1_")
        # stage-7 b-2
        self.clear_count()
        x1 = self.resnet(x1, 32, 104, strides=1, is_skip=True, prefix="s7_b2_")
        # stage-7 b-3
        self.clear_count()
        x1 = self.resnet(x1, 32, 104, strides=1, is_skip=True, prefix="s7_b3_")
        # stage-7 b-4
        self.clear_count()
        x1 = self.resnet(x1, 32, 104, strides=1, is_skip=True, prefix="s7_b4_")
        x1 = UpSampling2D(size=(2, 2))(x1)


        x2 = self.conv2d(i1, filters=104, strides=1, activation=None, name="conv", use_bias=False)
        # merge x1 and x2
        i2 = self.add([x1, x2], name="combine")

        # header 1
        x3 = self.conv2d(i2, filters=104, strides=1, name="P2_conv")
        x3 = self.conv2d(x3, filters=24, strides=1, name="P2_conv_0")
        self.clear_count()
        self.update_count()
        x3 = self.upconv(x3, filters=16, strides=1, prefix="P2_")
        x3 = self.upconv(x3, filters=16, strides=1, prefix="P2_")
        x3 = self.upconv(x3, filters=24, strides=1, name="P2_conv_3_m1")
        x3 = self.conv2d(x3, filters=32, strides=1, prefix="P2_")
        x3 = self.conv2d(x3, filters=1,  strides=1, prefix="P2_", use_bias=False)

        # header 2
        self.clear_count()
        x4 = self.conv2d(i2, filters=24, strides=1, prefix="P1_")
        x4 = self.upconv(x4, filters=24, strides=1, prefix="P1_")
        x4 = self.upconv(x4, filters=8, strides=1,  prefix="P1_")
        x4 = self.upconv(x4, filters=8, strides=1,  prefix="P1_")
        x4 = self.conv2d(x4, filters=8, strides=1,  name="P1_conv_3_paf")
        x4 = self.conv2d(x4, filters=2, strides=1,  prefix="P1_", use_bias=False, activation=None)

        # return the logits
        return Model(inputs=input, outputs=[x3, x4])

def mtnet_model(input_shape):
    net = Mtnet()
    return net.gen(input_shape)

# Create an instance of the model
def gen_net():
    config = g_config.get_cfg()
    format = config.keras_format
    if "first" in format:
        inp_shape = (None, config.input_c, config.input_h, config.input_w)
    else:
        inp_shape = (None, config.input_h, config.input_w, config.input_c)
    print("input shape:{}, keras data format:{}".format(inp_shape, format))

    net = mtnet_model(inp_shape)
    net.build(input_shape=inp_shape)
    net.summary()
    return net

def gen_optimizer():
    optimizer = tf.keras.optimizers.Adam()
    return optimizer

def compute_loss(logits, labels, name='loss'):
    with tf.name_scope(name):
        #prob = tf.nn.softmax(logits)
        label = tf.broadcast_to(labels[0], logits[0].shape)
        loss0 = tf.keras.losses.MeanSquaredError()(
                label, logits[0])
        loss0 = tf.reduce_mean(loss0)

        label = tf.broadcast_to(labels[1], logits[1].shape)
        loss1 = tf.keras.losses.MeanSquaredError()(
                label, logits[1])
        loss1 = tf.reduce_mean(loss1)

    return loss0+loss1

def compute_acc(logits, labels, name='acc'):
    with tf.name_scope(name):
        accuracy = tf.zeros(shape=(1,), dtype=tf.float32)
    return accuracy

def compute_predict(logits, name='predict'):
    with tf.name_scope(name):
        pred = tf.zeros(shape=(1,), dtype=tf.float32)
    return pred

def compute_predict_proc(logits, x_raw, y_raw):
    pass
    return None

def metrics_eval(predict, labels, name="metrics"):
    with tf.name_scope(name):
        y = tf.cast(tf.argmax(labels, axis=-1), tf.int32)
        metrics, metrics_op = tf.metrics.accuracy(labels=y, predictions=predict)
    return metrics, metrics_op

def train_op(loss, global_steps, opt_name, **kwargs):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = _get_optimizer(opt_name, kwargs)
        op = optimizer.minimize(loss, global_step=global_steps)
    return op


def setup_summary(loss, acc):
    #summary_loss = tf.summary.scalar('loss', loss)
    #summary_acc = tf.summary.scalar('acc', acc)
    #return tf.summary.merge([summary_loss, summary_acc])
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('acc', acc)
    return tf.summary.merge_all()

def _get_optimizer(opt_name, params):
    if opt_name == 'adam':
        return tf.train.AdamOptimizer(params['lr'])
    elif opt_name == 'adadelta':
        return tf.train.AdadeltaOptimizer(params['lr'])
    elif opt_name == 'sgd':
        return tf.train.GradientDescentOptimizer(params['lr'])
    elif opt_name == 'momentum':
        return tf.train.MomentumOptimizer(params['lr'], params['momentum'])
    elif opt_name == 'rms':
        return tf.train.RMSPropOptimizer(params['lr'])
    elif opt_name == 'adagrad':
        return tf.train.AdagradOptimizer(params['lr'])
    else:
        print('error')

#test
if __name__ == '__main__':
    config_dict = {
        "num_class": 10,
        "input_h": 448,
        "input_w": 1024,
        "input_c": 1,
        "batch_size": 5,
        "num_iter_per_epoch":2000000,
    }
    config_dict.update(keras_format=tf.keras.backend.image_data_format())
    config = Bunch(config_dict)
    g_config.__init(config)

    class Train():
        def __init__(self):
            self.net = gen_net()

            self.optimizer = gen_optimizer()

            self.accuracy = (lambda logits, labels: compute_acc(logits, labels))

            self.loss = (lambda logits, labels: compute_loss(logits, labels))

        def train_step(self, model, optimizer, images, labels):
            with tf.GradientTape() as tape:
                logits = model(images, training=True)
                loss = self.loss(logits, labels)
            # compute gradient
            grads = tape.gradient(loss, model.trainable_variables)
            # update to weights
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            acc = self.accuracy(logits, labels)
            # loss and accuracy is scalar tensor
            return logits, loss, acc

        def test_step(self, model, images, labels):
            logits = model(images, training=False)
            loss = self.loss(logits, labels)
            acc = self.accuracy(logits, labels)
            # loss and accuracy is scalar tensor
            return logits, loss, acc

        def save(self):
            path = os.path.dirname(os.path.abspath(__file__))
            print("Saving model...")
            self.net.save(path + "/saved_model")
            print("==== PB Model saved ===")
            self.net.save(path + "/model.h5", save_format='h5')
            print("==== h5 Model saved ====")

    train = Train()

    if "first" in g_config.get_cfg().keras_format:
        image = tf.random.normal(shape=[2, 1, 448, 1024], dtype=tf.float32)
        label1 = tf.ones(shape=[1, 112, 256], dtype=tf.float32)
        label2 = tf.ones(shape=[2, 112, 256], dtype=tf.float32)
    else:
        image = tf.random.normal(shape=[2, 448, 1024, 1], dtype=tf.float32)
        label1 = tf.ones(shape=[112, 256, 1], dtype=tf.float32)
        label2 = tf.ones(shape=[112, 256, 2], dtype=tf.float32)
    print(image.numpy().shape)
    label = [label1, label2]
    for i in range(20):
        _, loss, acc = train.train_step(train.net, train.optimizer, image, label)
        print("train loss:{}, acc:{}".format(loss.numpy(), acc.numpy()))

        if i%3 == 0:
            _, loss, acc = train.test_step(train.net, image, label)
            print("test loss:{}, acc:{}".format(loss.numpy(), acc.numpy()))

    train.save()
