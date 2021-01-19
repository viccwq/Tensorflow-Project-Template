# https://blog.csdn.net/xinxiang7/article/details/110822493
# https://github.com/ysh329/deep-learning-model-convertor
# http://caffe.berkeleyvision.org/tutorial/layers.html
# https://blog.csdn.net/github_37973614/article/details/81810327
import os
import sys
import time
import numpy as np
import caffe
import tensorflow as tf
import tensorflow.keras as keras
import cv2
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Activation, BatchNormalization, Flatten, Dense
from tensorflow.keras import Input, Model, Sequential
from tools.convert import convert

net  = "mtnet" #lenet
file = os.path.dirname(os.path.abspath(__file__)) + "/../experiments/" + net + "/model.h5"
caffe_proto='caffe.prototxt'
caffe_weights='caffe.caffemodel'
keras_format = keras.backend.image_data_format()

def keras2caffe(model, caffe_proto='caffe.prototxt', caffe_weights='caffe.caffemodel'):
    keras_model = keras.models.load_model(model, custom_objects=None)
    keras_model.summary()
    input_shape     = [1, keras_model.input.shape[3], keras_model.input.shape[1],keras_model.input.shape[2]]
    output_shape    = None
    convert(keras_model, keras_format, caffe_proto, caffe_weights)

def caffe_verification(model, caffe_proto='caffe.prototxt', caffe_weights='caffe.caffemodel'):
    c_net = caffe.Net(caffe_proto, caffe_weights, caffe.TEST)
    k_net = keras.models.load_model(model, custom_objects=None)

    if "lenet" in net:
        img = cv2.imread('mnist_3.bmp')
        #img = cv2.resize(img, (224, 224))
        #img = img[..., ::-1]  # RGB 2 BGR
        img = img[:,:,0]
    elif "mtnet" in net:
        img = np.random.rand(448, 1024) * 255.0
        img = img.astype(np.int)
    else:
        return


    data = np.array(img, dtype=np.float32)

    pred_caffe = {}
    pred_tf = {}
    #caffe inference
    c_data = data / 1.0
    c_data = np.expand_dims(c_data, (0,1))
    c_net.blobs['data'].data[...] = c_data
    out = c_net.forward()
    for (d, x) in out.items():
        print(d)
        pred_caffe[d] = np.squeeze(x)

    #keras inference
    k_data = data / 255.0
    k_data = tf.Variable(k_data, trainable=False)
    k_data = tf.expand_dims(k_data, axis=0)
    if "first" in keras_format:
        k_data = tf.expand_dims(k_data, axis=1)
    else:
        k_data = tf.expand_dims(k_data, axis=-1)

    out = k_net.call(k_data)
    for i, x in enumerate(out):
        print(i)
        pred_tf[i] = np.squeeze(x.numpy())

    if "lenet" in net:
        def compare(pred):
            prob = np.max(pred)
            cls = pred.argmax()
            print(pred, prob)
            print(cls)
        pred = pred_caffe["dense_1"]
        compare(pred)
        pred = pred_tf[0]
        compare(pred)
    elif "mtnet" in net:
        def compare(pred):
            prob = np.max(pred)
            print(pred[0, 0:10], pred[0, -10::])
            #print(pred[0, 0, 0:10], pred[0, 0, -10::])
            #print(pred[-1, 0, 0:10], pred[0, 0, -10::])
            print(pred.shape)
            print(prob)
            print("==========================")

        pred0 = pred_caffe["P2_conv_4"]
        compare(pred0)
        pred1 = pred_tf[0]
        if "first" in keras_format:
            pass
        else:
            pred1 = pred1.transpose(0,1)
        compare(pred1)
    else:
        pass

def main():
    keras2caffe(file)
    caffe_verification(file)

if __name__ == '__main__':
    main()