from numpy import ndarray

import tensorflow as tf
import tensorflow.contrib.layers as tcl
import time
from tensorflow.contrib.framework import arg_scope
import scipy.io as sio
import numpy as np
import os


from Unet_structure import conv_layer, conv_layer_transpose, conv2d, conv_layer_no_Var
from tensorflow.python.ops import init_ops
from tensorflow.contrib.layers.python.layers import initializers

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
    # random.shuffle(alist)
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




    return data


if __name__ == '__main__':
    version = '1020_v1'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    save_num = 0
    batch_size = 10
    dir_test_data = os.path.join(
        'G:\\CGT\\Matrix_completion\\matlab\\data\\test_data\single\\56mic_3_-09d_different_measurement_signals_1100Hz_1100Hz_6000Hz\\25')

    iters_epoch_train = np.int64(len(os.listdir(dir_test_data)) / batch_size)

    learning_rate = 0.01

    model_name = 'matrix_completion'
    save_dir_fig = os.path.join('./save_fig/unet/mixture', version)

    save_dir_train = os.path.join('./save_net_train/Unet/mixture', version)
    Create_Folder(save_dir_train)
    Create_Folder(save_dir_fig)
    save_train_txt_dir = os.path.join(dir_test_data, 'save_single_txt.txt')

    mat_save_path = './mat'
    Create_Folder(mat_save_path)

    gra1 = tf.Graph()
    config = tf.compat.v1.ConfigProto()


    Frob_norm_label = np.ones([batch_size, 56*3, 56*3, 1], dtype=np.complex)
    Frob_norm_input = np.ones([batch_size, 56*3, 56*3, 1], dtype=np.complex)


    print('paramenters initialize is ok')

    with gra1.as_default():

        input_data_amplitude = tf.compat.v1.placeholder(tf.float32, shape=[batch_size, 56*3, 56*3, 1])
        input_data_phase = tf.compat.v1.placeholder(tf.float32, shape=[batch_size, 56*3, 56*3, 1])

        training_flag = tf.compat.v1.placeholder(tf.bool)



        if training_flag is False:
            reuse_flag = True
        else:
            reuse_flag = False

        with tf.compat.v1.variable_scope("amplitude", reuse=reuse_flag):
            with tf.compat.v1.variable_scope('block1'):
                net_1_amplitude = conv_layer_no_Var(input_data_amplitude, out_deep=64, stride=1, istraining=training_flag)
                net_1_amplitude_pl = tf.nn.max_pool2d(net_1_amplitude, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                                      padding='SAME', name='pooling1')#60
            with tf.compat.v1.variable_scope('block2'):
                net_2_amplitude = conv_layer_no_Var(net_1_amplitude_pl, out_deep=128, stride=1, istraining=training_flag)
                net_2_amplitude_pl = tf.nn.max_pool2d(net_2_amplitude, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                                      padding='SAME', name='pooling2')#30
            with tf.compat.v1.variable_scope('block3'):
                net_3_amplitude = conv_layer_no_Var(net_2_amplitude_pl, out_deep=256, stride=1, istraining=training_flag)
                net_3_amplitude_pl = tf.nn.max_pool2d(net_3_amplitude, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                                      padding='SAME', name='pooling3')#15
            with tf.compat.v1.variable_scope('block4'):
                net_4_amplitude = conv_layer_no_Var(net_3_amplitude_pl, out_deep=512, stride=1, istraining=training_flag)
                net_4_amplitude_pl = tf.nn.max_pool2d(net_4_amplitude, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                                      padding='SAME', name='pooling4') # 8

            with tf.compat.v1.variable_scope('block1_trans'):
                net_1_amplitude_u = conv_layer(net_4_amplitude_pl, out_deep=1024, stride=1, istraining=training_flag)
                net_1_amplitude_trans = conv_layer_transpose(net_1_amplitude_u, out_deep=512, stride=2,
                                                             istraining=training_flag)# 15
            with tf.compat.v1.variable_scope('block2_trans'):
                unet_input = tf.concat([net_4_amplitude, net_1_amplitude_trans], axis=3)
                net_2_amplitude_u = conv_layer(unet_input, out_deep=512, stride=1, istraining=training_flag)
                net_2_amplitude_trans = conv_layer_transpose(net_2_amplitude_u, out_deep=256, stride=2,
                                                             istraining=training_flag)  # 30
            with tf.compat.v1.variable_scope('block3_trans'):
                unet_input = tf.concat([net_3_amplitude, net_2_amplitude_trans], axis=3)
                net_3_amplitude_u = conv_layer(unet_input, out_deep=256, stride=1, istraining=training_flag)
                net_3_amplitude_trans = conv_layer_transpose(net_3_amplitude_u, out_deep=128, stride=2,
                                                             istraining=training_flag)  # 60
            with tf.compat.v1.variable_scope('block4_trans'):
                unet_input = tf.concat([net_2_amplitude, net_3_amplitude_trans], axis=3)
                net_4_amplitude_u = conv_layer(unet_input, out_deep=128, stride=1, istraining=training_flag)
                net_4_amplitude_trans = conv_layer_transpose(net_4_amplitude_u, out_deep=64, stride=2,
                                                             istraining=training_flag)  # 56*3
            with tf.compat.v1.variable_scope('block5_trans'):
                unet_input = tf.concat([net_1_amplitude, net_4_amplitude_trans], axis=3)
                net_5_amplitude_u = conv_layer(unet_input, out_deep=64, stride=1, istraining=training_flag)

            net_out_amplitude = conv2d(net_5_amplitude_u, out_num=1, kernel_size=1, stride_kernel=1,
                                       activation_fn=tf.nn.leaky_relu, padding='SAME',
                                       weights_initializer=initializers.xavier_initializer(),
                                       bias_initializer=init_ops.zeros_initializer(), is_training=training_flag)

        with tf.compat.v1.variable_scope("phase", reuse=reuse_flag):
            with tf.compat.v1.variable_scope('block1'):
                net_1_phase = conv_layer_no_Var(input_data_phase, out_deep=64, stride=1, istraining=training_flag)
                net_1_phase_pl = tf.nn.max_pool2d(net_1_phase, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                                  padding='SAME', name='pooling1')  # 60
            with tf.compat.v1.variable_scope('block2'):
                net_2_phase = conv_layer_no_Var(net_1_phase_pl, out_deep=128, stride=1, istraining=training_flag)
                net_2_phase_pl = tf.nn.max_pool2d(net_2_phase, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                                  padding='SAME', name='pooling2')  # 30
            with tf.compat.v1.variable_scope('block3'):
                net_3_phase = conv_layer_no_Var(net_2_phase_pl, out_deep=256, stride=1, istraining=training_flag)
                net_3_phase_pl = tf.nn.max_pool2d(net_3_phase, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                                  padding='SAME', name='pooling3')  # 15
            with tf.compat.v1.variable_scope('block4'):
                net_4_phase = conv_layer_no_Var(net_3_phase_pl, out_deep=512, stride=1, istraining=training_flag)
                net_4_phase_pl = tf.nn.max_pool2d(net_4_phase, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                                  padding='SAME', name='pooling4')  # 8

            with tf.compat.v1.variable_scope('block1_trans'):
                net_1_phase_u = conv_layer(net_4_phase_pl, out_deep=1024, stride=1, istraining=training_flag)
                net_1_phase_trans = conv_layer_transpose(net_1_phase_u, out_deep=512, stride=2,
                                                         istraining=training_flag)  # 15
            with tf.compat.v1.variable_scope('block2_trans'):
                unet_input = tf.concat([net_4_phase, net_1_phase_trans], axis=3)
                net_2_phase_u = conv_layer(unet_input, out_deep=512, stride=1, istraining=training_flag)
                net_2_phase_trans = conv_layer_transpose(net_2_phase_u, out_deep=256, stride=2,
                                                         istraining=training_flag)  # 30
            with tf.compat.v1.variable_scope('block3_trans'):
                unet_input = tf.concat([net_3_phase, net_2_phase_trans], axis=3)
                net_3_phase_u = conv_layer(unet_input, out_deep=256, stride=1, istraining=training_flag)
                net_3_phase_trans = conv_layer_transpose(net_3_phase_u, out_deep=128, stride=2,
                                                             istraining=training_flag)  # 60
            with tf.compat.v1.variable_scope('block4_trans'):
                unet_input = tf.concat([net_2_phase, net_3_phase_trans], axis=3)
                net_4_phase_u = conv_layer(unet_input, out_deep=128, stride=1, istraining=training_flag)
                net_4_phase_trans = conv_layer_transpose(net_4_phase_u, out_deep=64, stride=2,
                                                             istraining=training_flag)  # 56*3
            with tf.compat.v1.variable_scope('block5_trans'):
                unet_input = tf.concat([net_1_phase, net_4_phase_trans], axis=3)
                net_5_phase_u = conv_layer(unet_input, out_deep=64, stride=1, istraining=training_flag)

            net_out_phase = conv2d(net_5_phase_u, out_num=1, kernel_size=1, stride_kernel=1,
                                       activation_fn=tf.nn.leaky_relu, padding='SAME',
                                       weights_initializer=initializers.xavier_initializer(),
                                       bias_initializer=init_ops.zeros_initializer(), is_training=training_flag)

        net_out_amplitude = tf.add(tf.transpose(net_out_amplitude, perm=[0, 2, 1, 3]), net_out_amplitude) / 2
        net_phase_differ = tf.add(tf.transpose(net_out_phase, perm=[0, 2, 1, 3]), net_out_phase) / 2

        net_out_phase = net_out_phase - net_phase_differ
        diag_out = tf.linalg.diag_part(tf.squeeze(net_out_amplitude))  # batch_size，56*3,1
        diag_out = tf.expand_dims(diag_out, axis=2)
        diag_value = tf.sqrt(tf.abs(tf.matmul(diag_out, tf.transpose(diag_out, perm=[0, 2, 1]))))
        constrain_value = diag_value / tf.squeeze(net_out_amplitude)
        net_out_amplitude = tf.multiply(net_out_amplitude, tf.expand_dims(constrain_value, axis=3))
        output_data = tf.complex(net_out_amplitude * tf.cos(net_out_phase), net_out_amplitude * tf.sin(net_out_phase))

    print('tensorflow graph initialize is ok')


    with tf.compat.v1.Session(graph=gra1, config=config) as sess:
        saver = tf.compat.v1.train.Saver(max_to_keep=50)


        # reload the saved net
        saver.restore(sess, os.path.join(save_dir_train, 'matrix_completion-0'))

        # tf.compat.v1.global_variables_initializer().run()
        print('tensorflow graph weights initialize is ok')

        for epoch in range(1):
            learning_rate = 0.05 * np.exp(-epoch / 100)
            print('learning_rate is ', learning_rate)

            yield_list = create_batch_list(dir_test_data, batch_size)

            for iters in range(0, iters_epoch_train):

                batch_list = next(yield_list)
                print(batch_list)

                mat_data = create_batch_data(dir_test_data, batch_list)  # alldata shape 32,1,434,13

                net_input_data = np.reshape(mat_data, [batch_size, 56*3, 56*3])


                ##################################data normalization###########################################################################


                data_real = np.real(net_input_data)
                data_imag = np.imag(net_input_data)
                phase_input = np.expand_dims(np.arctan2(data_imag, data_real), axis=3)
                processed_input = data_real + 1j * data_imag

                processed_input_norm = np.linalg.norm(processed_input, axis=(1, 2))
                for i in range(0, batch_size):
                    normdata = np.expand_dims((processed_input[i, :, :] / processed_input_norm[i]), axis=2)
                    Frob_norm_input[i, :, :, :] = normdata * 50

                amplitude_input = np.abs(Frob_norm_input)

                amplitude_in = amplitude_input
                phase_in = phase_input
                training_flag_placeholder = True

                ###########running time########################
                starttime = time.time()
                training_epoch_am, training_epoch_ph, output_data_sess = sess.run(
                    [net_out_amplitude, net_out_phase, output_data],
                    feed_dict={input_data_amplitude: amplitude_in,
                               input_data_phase: phase_in,
                               # label_amplitude: amplitude_label,
                               # label_phase: phase_label,
                               training_flag: training_flag_placeholder})

                endtime = time.time()
                print('running time:', (endtime - starttime))
                # print("epoch is", epoch, "iters is", iters, "loss is : ", loss_train)
                ###################################################################################################################
                out_load = output_data_sess
                # input_load = Frob_norm_label
                bat = 0
                for bat_dir in batch_list:
                    # date_chron_R = amplitude_label * np.cos(phase_label) + 1j * amplitude_label * np.sin(
                    #     phase_label)
                    # date_chron_R_sav = np.squeeze(date_chron_R[bat, :, :, :])
                    output_data_sess_sav = np.squeeze(output_data_sess[bat, :, :, :])
                    # print('date_chron_R shape is ',date_chron_R_sav.shape)
                    # print(bat_dir)
                    date_nonchron_R_sav = np.squeeze(Frob_norm_input[bat, :, :, :])
                    # Loc_S_sav = source_localization[bat, :]
                    sio.savemat(os.path.join(mat_save_path, bat_dir), {'date_nonchron_corr': date_nonchron_R_sav,
                                                                       'date_nonchron_Net': output_data_sess_sav,})
                                                                       # 'date_synchron_corr': date_chron_R_sav,
                                                                       # 'Loc_S': Loc_S_sav})


                    bat = bat + 1










