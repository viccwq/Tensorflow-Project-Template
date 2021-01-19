import os
import sys
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Activation, BatchNormalization, Flatten, Dense, Concatenate, Permute
import utils.config_val as g_config

def FlattenCaffe(inp):
    keras_format = g_config.get_cfg().keras_format
    shape = inp.shape
    # finish the transpose if working in channel last mode
    # the layer Permute should be skipped during frame work conversion
    if "first" in keras_format:
        outp = inp
    else:
        outp = Permute((3, 1, 2), input_shape=(shape[1], shape[2], shape[3]))(inp)
    # always using the channel_last mode
    return  Flatten(data_format="channels_last")(outp)


def ConcatCaffe(inps):
    keras_format = g_config.get_cfg().keras_format
    if "first" in keras_format:
        outp = Concatenate(axis=1)(inps)
    else:
        outp = Concatenate(axis=3)(inps)
    return outp