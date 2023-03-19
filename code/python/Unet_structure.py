

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from numpy import ndarray
from tensorflow.contrib.framework import arg_scope
from tensorflow.python.ops import init_ops
from tensorflow.contrib.layers.python.layers import initializers

import tensorflow.contrib.layers as tcl
import tensorflow as tf

import numpy as np
import os
import random
import scipy.io as sio

class array_to_matrix(ndarray):
    @property
    def H(self):
        return self.conj().T

def batch_norm_wrapper(x, training, scope):
    with arg_scope([tcl.batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True):
        return tf.cond(training,
                       lambda: tcl.batch_norm(inputs=x, is_training=training, reuse=None),
                       lambda: tcl.batch_norm(inputs=x, is_training=training, reuse=True))


def shuffle_every_epoch(DIR):  # 列出当前目录DIR下的所有文件，并随机排序
    '''
    :param DIR: 输入要查看的目录路径
    :return: 返回当前路径下所有经过打乱的文件名。
    '''
    alist = os.listdir(DIR)
    random.shuffle(alist)
    return alist


def Create_Folder(fd):
    """
    :param fd: 输入文件夹的名字，如果没有则创建（可以是路径）
    :return: None
    """
    if not os.path.exists(fd):  # os.path.exists如果path存在，返回True；如果path不存在，返回False
        os.makedirs(fd)


def create_batch_list(DIR, batch_size):
    '''
    :param DIR: 目录路径
    :param batch_size:
    :return:
    '''
    alist = shuffle_every_epoch(DIR)
    # print(len(alist))
    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size > len(alist):  # basailuona len(alist) = 965 zonggong 5603
            batch_count = 0
        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        yield alist[start:end]  # yield 是一个类似 return 的关键字，迭代一次遇到yield时就返回yield后面(右边)的值。
        # 重点是：下一次迭代时，从上一次迭代遇到的yield后面的代码(下一行)开始执行。而且yeild与next函数（next(get_mini_batch)）搭配使用


def load_MATLAB_data(path):
    # data = h5py.File(path)
    data = sio.loadmat(path)

    return data


def create_batch_data(DIR, Batch_list):
    '''
    :param file:
    :param DIR: 目录路径
    :param Batch_list: 生成的batch列表
    :param Data_type: 选择要使用的数据，是mfcc还是fbank
    :return:返回由batchlist产生的数据data和label
    '''
    data = []
    batch_label = []
    localization = []
    for file in Batch_list:
        # print(file)
        npz_file = load_MATLAB_data(os.path.join(DIR, file))
        data.append(npz_file["date_nonchron_corr"])
        batch_label.append(npz_file['date_synchron_corr'])
        localization.append(npz_file['Loc_S'][:, 0])
        # print(file)


    return data, batch_label, localization


def conv2d(x, out_num, kernel_size, stride_kernel=1, activation_fn=None, decay=0.999, normalizer_params=None,
           weights_initializer=tf.random_normal_initializer(-0.01, 0.01),
           bias_initializer=None, padding='SAME', is_training=True, scope='conv'):
    if is_training is False:
        reuse_flag = True
    else:
        reuse_flag = False
        # print('istraining is True')

    x_shape = x.shape

    with tf.compat.v1.variable_scope("conv", reuse=reuse_flag):
        weights = tf.compat.v1.get_variable(scope + 'weights',
                                            [kernel_size, kernel_size, x_shape[3], out_num],
                                            initializer=weights_initializer)
    if bias_initializer is None:

        layer_out = tf.nn.conv2d(x, weights, strides=[1, stride_kernel, stride_kernel, 1], padding=padding)

    else:

        with tf.compat.v1.variable_scope("conv", reuse=reuse_flag):
            bias = tf.compat.v1.get_variable(scope + 'bias', out_num, initializer=bias_initializer)
        layer_out = tf.nn.conv2d(x, weights, strides=[1, stride_kernel, stride_kernel, 1], padding=padding)

        layer_out = tf.nn.bias_add(layer_out, bias, data_format='NHWC')

    if activation_fn is not None:
        layer_out = activation_fn(layer_out, alpha=0.1, name='activation')

    if normalizer_params is not None:
        layer_out = normalizer_params(layer_out, decay=decay, is_training=is_training, scope=scope + 'batch_norm')

    return layer_out


def conv2d_transpose(x, out_num, kernel_size, stride_kernel=1, activation_fn=None, decay=0.999,
                     normalizer_params=None, weights_initializer=tf.random_normal_initializer(-0.01, 0.01),
                     bias_initializer=None, padding='SAME', is_training=True, scope='conv_trans'):
    if is_training is False:
        reuse_flag = True
    else:
        reuse_flag = False

    x_shape = x.shape
    with tf.compat.v1.variable_scope("conv", reuse=reuse_flag):
        weights = tf.compat.v1.get_variable(scope + 'weights',
                                            [kernel_size, kernel_size, out_num, x_shape[3]],
                                            initializer=weights_initializer)
    if bias_initializer is None:

        if stride_kernel * x_shape[2] == 12:
            out_size = 11
        elif stride_kernel * x_shape[2] == 14:
            out_size = 13
        elif stride_kernel * x_shape[2] == 16:
            out_size = 15
        elif stride_kernel * x_shape[2] == 26:
            out_size = 25
        elif stride_kernel * x_shape[2] == 50:
            out_size = 49
        elif stride_kernel * x_shape[2] == 22:
            out_size = 21
        else:
            out_size = stride_kernel * x_shape[2]

        layer_out = tf.nn.conv2d_transpose(x, weights,
                                           output_shape=[x_shape[0], out_size, out_size,
                                                         out_num],
                                           strides=[1, stride_kernel, stride_kernel, 1],
                                           padding=padding, data_format='NHWC')
    else:

        with tf.compat.v1.variable_scope("conv", reuse=reuse_flag):
            bias = tf.compat.v1.get_variable(scope + 'bias', out_num, initializer=bias_initializer)
        if stride_kernel * x_shape[2] == 12:
            out_size = 11
        elif stride_kernel * x_shape[2] == 14:
            out_size = 13
        elif stride_kernel * x_shape[2] == 16:
            out_size = 15
        elif stride_kernel * x_shape[2] == 26:
            out_size = 25
        elif stride_kernel * x_shape[2] == 50:
            out_size = 49
        elif stride_kernel * x_shape[2] == 22:
            out_size = 21
        else:
            out_size = stride_kernel * x_shape[2]
        # if stride_kernel * x_shape[2] == 36:
        layer_out = tf.nn.conv2d_transpose(x, weights,
                                               output_shape=[x_shape[0], out_size, out_size,
                                                             out_num],
                                               strides=[1, stride_kernel, stride_kernel, 1],
                                               padding=padding, data_format='NHWC')


        layer_out = tf.nn.bias_add(layer_out, bias, data_format='NHWC')

    if activation_fn is not None:
        layer_out = activation_fn(layer_out, alpha=0.1, name='activation')
    if normalizer_params is not None:
        layer_out = tcl.batch_norm(layer_out, decay=decay, is_training=is_training, scope=scope + 'batch_norm')

    return layer_out


def conv_layer_no_Var(x, out_deep=1, stride=1, istraining=True):
    with tf.compat.v1.variable_scope('conv1'):
        g = conv2d(x, out_num=out_deep, kernel_size=3, stride_kernel=stride,
                   activation_fn=tf.nn.leaky_relu, padding='SAME', normalizer_params=tcl.batch_norm,
                   weights_initializer=initializers.xavier_initializer(), is_training=istraining)

    with tf.compat.v1.variable_scope('conv2'):
        unit = conv2d(g, out_num=out_deep, kernel_size=3, stride_kernel=stride,
                    activation_fn=tf.nn.leaky_relu, padding='SAME', normalizer_params=tcl.batch_norm,
                    weights_initializer=initializers.xavier_initializer(), is_training=istraining)

    # unit = tf.layers.dropout(inputs=unit, rate=0.1, training=istraining, name='conv_dropout')
    return unit
def conv_layer(x, out_deep=1, stride=1, istraining=True):
    with tf.compat.v1.variable_scope('conv1'):
        g = conv2d(x, out_num=out_deep, kernel_size=7, stride_kernel=stride,
                   activation_fn=tf.nn.leaky_relu, padding='SAME', normalizer_params=tcl.batch_norm,
                   weights_initializer=initializers.xavier_initializer(),
                   bias_initializer=initializers.xavier_initializer(), is_training=istraining)

    with tf.compat.v1.variable_scope('conv2'):
        unit = conv2d(g, out_num=out_deep, kernel_size=7, stride_kernel=stride,
                    activation_fn=tf.nn.leaky_relu, padding='SAME', normalizer_params=tcl.batch_norm,
                    weights_initializer=initializers.xavier_initializer(),
                    bias_initializer=init_ops.zeros_initializer(), is_training=istraining)

    # unit = tf.layers.dropout(inputs=unit, rate=0.1, training=istraining, name='conv_dropout')
    return unit



def block_fc(x ,out_num=1, repeat_times=1, activation_fn=tf.nn.leaky_relu,
             weights_initializer=initializers.xavier_initializer(),
             biases_initializer=init_ops.zeros_initializer(),
             istraining=True, scope="blocks"):
    net = x
    if istraining is False:
        reuse_flag = True
    else:
        reuse_flag = False

    for i in range(1, repeat_times+1):
        with tf.compat.v1.variable_scope(scope + np.str(i)):
            net = tcl.fully_connected(net, out_num, activation_fn=activation_fn,
                                      weights_initializer=weights_initializer,
                                      biases_initializer=biases_initializer,
                                      reuse=reuse_flag,
                                      trainable=istraining,
                                      scope='fc')

    return net





def conv_layer_transpose(inputs, out_deep=1, stride=1, istraining=True):
    with tf.compat.v1.variable_scope('conv_trans_1'):
        unit = conv2d_transpose(inputs, out_num=out_deep, kernel_size=2, stride_kernel=stride,
                                activation_fn=tf.nn.leaky_relu, padding='SAME', normalizer_params=tcl.batch_norm,
                                 weights_initializer=initializers.xavier_initializer(),
                                 bias_initializer=init_ops.zeros_initializer(), is_training=istraining)
    # unit = tf.layers.dropout(inputs=unit, rate=0.1, training=istraining, name='conv_trans_dropout')

    return unit



def loss(logits_out, label, scope="loss"):
    with tf.compat.v1.variable_scope(scope):
        loss_ce = tf.reduce_mean(
            tf.compat.v1.losses.mean_squared_error(predictions=logits_out, labels=label))

        tf.compat.v1.add_to_collection(scope, loss_ce)
        loss_mse = tf.add_n(tf.compat.v1.get_collection(scope))
        tf.compat.v1.summary.scalar(scope, loss_mse)

    return loss_mse


def training_Optimizer(loss_all, learning_rate=0.005):
    train_adam = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    train = train_adam.minimize(loss_all)
    return train


