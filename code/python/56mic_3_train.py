
from numpy import ndarray

import tensorflow as tf
import tensorflow.contrib.layers as tcl

import numpy as np
import os

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from Unet_structure import Create_Folder, create_batch_list, load_MATLAB_data, create_batch_data
from Unet_structure import loss, training_Optimizer, conv_layer, conv_layer_transpose, conv2d, conv_layer_no_Var
from tensorflow.python.ops import init_ops
from tensorflow.contrib.layers.python.layers import initializers


class array_to_matrix(ndarray):
    @property
    def H(self):
        return self.conj().T


if __name__ == '__main__':
    version = '1020_v1'
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    dx = 0.03
    dy = 0.03
    zs = 0.6
    save_num = 0
    batch_size = 10
    dir_train_data = os.path.join('G:\\CGT\\Matrix_completion\\matlab\\data\\train_data\\single\\56mic_3_-09d\\25')
    iters_epoch_train = int(len(os.listdir(dir_train_data)) / batch_size)-1

    Num_Elements = 56*3


    Number_source = 1

    zs = 0.6
    fig = 0
    c = 343
    loss_test = 0
    localization_error = 0

    learning_rate = 0.01
    loss_train_recoder = 2

    model_name = 'matrix_completion'
    save_dir_fig = os.path.join('./save_fig/unet/mixture', version)

    save_dir_train = os.path.join('./save_net_train/Unet/mixture', version)
    Create_Folder(save_dir_train)
    Create_Folder(save_dir_fig)
    save_train_txt_dir = os.path.join(save_dir_train, 'save_single_txt.txt')

    gra1 = tf.Graph()
    config = tf.compat.v1.ConfigProto()

    x = np.arange(-2, 0, dx)
    y = np.arange(0, 2, dy)

    [Loc_scan_x, Loc_scan_y] = np.meshgrid(x, y)
    [ny, nx] = np.shape(Loc_scan_x)
    Frob_norm_label = np.ones([batch_size, 56*3, 56*3, 1], dtype=np.complex)
    Frob_norm_input = np.ones([batch_size, 56*3, 56*3, 1], dtype=np.complex)

    output_MUSIC = np.zeros([ny, nx])
    broadband_output_MUSIC = np.zeros([50, ny, nx])

    x_s = 0
    y_s = 0

    output_MUSIC = np.zeros([ny, nx], dtype=np.float)

    output_MUSIC_ori = np.zeros([ny, nx], dtype=np.float)
    broadband_output_MUSIC = np.zeros([50, ny, nx])

    print('paramenters initialize is ok')

    with gra1.as_default():

        input_data_amplitude = tf.compat.v1.placeholder(tf.float32, shape=[batch_size, 56*3, 56*3, 1])
        input_data_phase = tf.compat.v1.placeholder(tf.float32, shape=[batch_size, 56*3, 56*3, 1])
        label_amplitude = tf.compat.v1.placeholder(tf.float32, shape=[batch_size, 56*3, 56*3, 1])
        label_phase = tf.compat.v1.placeholder(tf.float32, shape=[batch_size, 56*3, 56*3, 1])
        training_flag = tf.compat.v1.placeholder(tf.bool)

        if training_flag is False:
            reuse_flag = True
        else:
            reuse_flag = False

        with tf.compat.v1.variable_scope("amplitude", reuse=reuse_flag):
            with tf.compat.v1.variable_scope('block1'):
                net_1_amplitude = conv_layer_no_Var(input_data_amplitude, out_deep=64, stride=1,
                                                    istraining=training_flag)
                net_1_amplitude_pl = tf.nn.max_pool2d(net_1_amplitude, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                                      padding='SAME', name='pooling1')  # 84

            with tf.compat.v1.variable_scope('block2'):
                net_2_amplitude = conv_layer_no_Var(net_1_amplitude_pl, out_deep=128, stride=1,
                                                    istraining=training_flag)
                net_2_amplitude_pl = tf.nn.max_pool2d(net_2_amplitude, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                                      padding='SAME', name='pooling2')  # 42
            with tf.compat.v1.variable_scope('block3'):
                net_3_amplitude = conv_layer_no_Var(net_2_amplitude_pl, out_deep=256, stride=1,
                                                    istraining=training_flag)
                net_3_amplitude_pl = tf.nn.max_pool2d(net_3_amplitude, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                                      padding='SAME', name='pooling3')  # 21
            with tf.compat.v1.variable_scope('block4'):
                net_4_amplitude = conv_layer_no_Var(net_3_amplitude_pl, out_deep=512, stride=1,
                                                    istraining=training_flag)
                net_4_amplitude_pl = tf.nn.max_pool2d(net_4_amplitude, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                                      padding='SAME', name='pooling4')  # 11

            with tf.compat.v1.variable_scope('block1_trans'):
                net_1_amplitude_u = conv_layer(net_4_amplitude_pl, out_deep=1024, stride=1, istraining=training_flag)
                net_1_amplitude_trans = conv_layer_transpose(net_1_amplitude_u, out_deep=512, stride=2,
                                                             istraining=training_flag)  # 21

            with tf.compat.v1.variable_scope('block2_trans'):
                # print('net_4_amplitude shape is ', net_4_amplitude.shape)
                # print('net_1_amplitude_trans shape is ', net_1_amplitude_trans.shape)
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
                                                             istraining=training_flag)  # 120
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
                                                  padding='SAME', name='pooling1')  # 59
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
                                                         istraining=training_flag)  # 59
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

        loss_amplitude = loss(net_out_amplitude, label_amplitude, scope='loss_amplitude')
        loss_phase = loss(net_out_phase, label_phase, scope='loss_phase')

        loss_all = loss_amplitude + loss_phase

        training_amplitude = training_Optimizer(loss_amplitude, learning_rate=learning_rate)
        training_phase = training_Optimizer(loss_phase, learning_rate=learning_rate)

        diag_out = tf.linalg.diag_part(tf.squeeze(net_out_amplitude))  # batch_size，56*3,1
        diag_out = tf.expand_dims(diag_out, axis=2)
        diag_value = tf.sqrt(tf.abs(tf.matmul(diag_out, tf.transpose(diag_out, perm=[0, 2, 1]))))
        constrain_value = diag_value / tf.squeeze(net_out_amplitude)
        net_out_amplitude = tf.multiply(net_out_amplitude, tf.expand_dims(constrain_value, axis=3))
        output_data = tf.complex(net_out_amplitude * tf.cos(net_out_phase), net_out_amplitude * tf.sin(net_out_phase))

    print('tensorflow graph initialize is ok')

    with tf.compat.v1.Session(graph=gra1, config=config) as sess:
        saver = tf.compat.v1.train.Saver(max_to_keep=50)
        tf.compat.v1.global_variables_initializer().run()
        print('tensorflow graph weights initialize is ok')
        Create_Folder(os.path.join('./tensorboard/Unet/mixture', version))
        merged = tf.summary.merge_all()
        summary_writer_train = tf.summary.FileWriter(os.path.join('./tensorboard/Unet/mixture', version), sess.graph)

        for epoch in range(10000):
            learning_rate = 0.05 * np.exp(-epoch / 100)
            print('learning_rate is ', learning_rate)

            yield_list = create_batch_list(dir_train_data, batch_size)

            for iters in range(0, iters_epoch_train):

                batch_list = next(yield_list)

                mat_data, mat_label, source_localization = create_batch_data(dir_train_data,
                                                                             batch_list)  # alldata shape 32,1,434,13

                net_input_data = np.reshape(mat_data, [batch_size, 56*3, 56*3])

                source_localization = np.reshape(source_localization, [batch_size, 2])
                net_aggressive_data = np.reshape(mat_label, [batch_size, 56*3, 56*3])

                ##################################label normalization###########################################################################

                aggres_data_norm = np.linalg.norm(net_aggressive_data, axis=(1, 2))

                for i in range(0, batch_size):
                    normdata = np.expand_dims((net_aggressive_data[i, :, :] / aggres_data_norm[i]), axis=2)
                    Frob_norm_label[i, :, :, :] = normdata * 100

                amplitude_label = np.abs(Frob_norm_label)

                phase_label = np.arctan2(np.imag(Frob_norm_label), np.real(Frob_norm_label))
                ########################################input normalization#######################################################################

                data_real = np.real(net_input_data)
                data_imag = np.imag(net_input_data)
                phase_input = np.expand_dims(np.arctan2(data_imag, data_real), axis=3)
                processed_input = data_real + 1j * data_imag

                processed_input_norm = np.linalg.norm(processed_input, axis=(1, 2))
                for i in range(0, batch_size):
                    normdata = np.expand_dims((processed_input[i, :, :] / processed_input_norm[i]), axis=2)
                    Frob_norm_input[i, :, :, :] = normdata * 100

                amplitude_input = np.abs(Frob_norm_input)

                _, _ = sess.run([training_amplitude, training_phase], feed_dict={input_data_amplitude: amplitude_input,
                                                                                 input_data_phase: phase_input,
                                                                                 label_amplitude: amplitude_label,
                                                                                 label_phase: phase_label,
                                                                                 training_flag: True})

                amplitude_in = amplitude_input
                phase_in = phase_input
                training_flag_placeholder = True
                training_epoch_am, training_epoch_ph, output_data_sess = sess.run(
                    [net_out_amplitude, net_out_phase, output_data],
                    feed_dict={input_data_amplitude: amplitude_in,
                               input_data_phase: phase_in,
                               label_amplitude: amplitude_label,
                               label_phase: phase_label,
                               training_flag: training_flag_placeholder})

                loss_train = sess.run(loss_all, feed_dict={input_data_amplitude: amplitude_in,
                                                           input_data_phase: phase_in,
                                                           label_amplitude: amplitude_label,
                                                           label_phase: phase_label,
                                                           training_flag: training_flag_placeholder})

                print("epoch is", epoch, "iters is", iters, "loss is : ", loss_train)
                # for bat_dir in batch_list:
                #     print(bat_dir[0:4])
                #######################################################################################################################
                if (epoch * iters_epoch_train * batch_size + iters) % 100 == 0:
                    value = tf.compat.v1.get_default_graph().get_tensor_by_name(
                        'amplitude/block1/conv1/conv/convweights:0')
                    print('amplitude/block1/conv1/conv/convweights:0', sess.run(value)[0, 0:5, 0:5, 0])
                    # output_data_sess = sess.run(output_data, feed_dict={input_data_amplitude: amplitude_input,
                    #                                                     input_data_phase: phase_input,
                    #                                                     label_amplitude: amplitude_label,
                    #                                                     label_phase: phase_label,
                    #                                                     training_flag: True})

                    print("net_out_amplitude_sess out is ", output_data_sess[0, 0:5, 0:5, 0])

                    print("label data is ", np.array(Frob_norm_label)[0, 0:5, 0:5, 0])

                    print("original data is ", np.array(Frob_norm_input)[0, 0:5, 0:5, 0])

                    # if loss_train < 2:
                    # if loss_train*2 < loss_train_recoder:
                    #     loss_train_recoder = loss_train
                    if save_num % 100 == 0:
                        saver.save(sess, os.path.join(save_dir_train, model_name),
                                   global_step=epoch * iters_epoch_train * batch_size + iters)
                        with open(save_train_txt_dir, 'a') as f:
                            f.write("train loss:%f, epoch:%s" %
                                    (loss_train, str(epoch * iters_epoch_train * batch_size + iters) + '\n'))
                    save_num = save_num + 1
                    fig = plt.figure(num=2, figsize=(18, 4))

                    plt.ion()  # open interactive mode
                    ###################################################

                    ###################################################################################################################
                    out_load = output_data_sess
                    input_load = Frob_norm_label
                    bat = 0
                    for bat_dir in batch_list:

                        if bat_dir[0] == '1' or bat_dir[0] == '2' or bat_dir[0] == '3' or bat_dir[0] == '4' or bat_dir[
                            0] == '5':
                            frequency_point = int(bat_dir[0:4])
                        elif bat_dir[0] == '6':
                            if bat_dir[1] == '1':
                                frequency_point = int(bat_dir[0:4])
                            else:
                                frequency_point = int(bat_dir[0:3])
                        else:
                            frequency_point = int(bat_dir[0:3])
                        ww = np.load(
                            os.path.join("..\matlab\scan_w\\56mic_3_06Z_-09str", str(frequency_point) + '.npy'))
                        print(frequency_point)
                        # print('out_load maximum element', np.max(out_load[bat, :, :]))
                        if (abs(np.max(out_load[bat, :, :])) < 10000000):

                            [D, V] = np.linalg.eig(np.squeeze(out_load[bat, :, :]))
                            DI = np.argsort(-abs(D))
                            Di = D[DI]
                            VI = V[:, DI]

                            Vn = VI[:, Number_source: Num_Elements].view(array_to_matrix)
                            inR_ev = np.dot(Vn, Vn.H)

                            for ii in range(nx):
                                for jj in range(ny):
                                    w = np.expand_dims(np.squeeze(ww[ii, jj, :]), axis=0).view(array_to_matrix)
                                    matmul = np.dot(w, inR_ev)
                                    matmul2 = np.dot(matmul, w.H)
                                    output_MUSIC[jj, ii] = np.squeeze(np.abs(1. / matmul2))
                            ax = plt.subplot(2, batch_size, bat + 1)
                            ##############calculate coordinates################

                            index_num = np.argmax(output_MUSIC) + 1
                            y_s_index = int(index_num / ny) + 1
                            y_s_index = int(ny - y_s_index)

                            x_s_index = index_num + 1 - int(index_num / ny) * ny
                            x_s_index = int(x_s_index)

                            if x_s_index >= 67:
                                x_s_index = 66
                            if y_s_index >= 67:
                                y_s_index = 66

                            x_s = x[x_s_index]
                            y_s = y[y_s_index]
                            plt.title(str(x_s)[0:4] + ',' + str(y_s)[0:4], fontsize=10)
                            if (bat == 0):
                                plt.ylabel('net_out', fontsize=10)

                            plt.imshow(output_MUSIC, extent=(
                                np.amin(Loc_scan_x), np.amax(Loc_scan_x), np.amin(Loc_scan_y), np.amax(Loc_scan_y)),
                                       cmap=cm.jet)
                        else:
                            ax = plt.subplot(2, batch_size, bat + 1)
                            plt.title('the max output contain Nan')
                        # plt.gca().invert_yaxis()
                        ###################################################################################################################

                        [D, V] = np.linalg.eig(np.squeeze(input_load[bat, :, :, :]))
                        DI = np.argsort(-abs(D))
                        Di = D[DI]
                        VI = V[:, DI]

                        Vn = VI[:, Number_source: Num_Elements].view(array_to_matrix)
                        inR_ev = np.dot(Vn, Vn.H)

                        for ii in range(nx):
                            for jj in range(ny):
                                w = np.expand_dims(np.squeeze(ww[ii, jj, :]), axis=0).view(array_to_matrix)
                                matmul = np.dot(w, inR_ev)
                                matmul2 = np.dot(matmul, w.H)
                                output_MUSIC_ori[jj, ii] = np.squeeze(np.abs(1. / matmul2))

                        ax1 = plt.subplot(2, batch_size, batch_size + bat + 1)
                        plt.title(
                            str(source_localization[bat, 0])[0:4] + ',' + str(2 - source_localization[bat, 1])[0:4],
                            fontsize=10)
                        plt.xlabel(str(frequency_point), fontsize=10)
                        if (bat == 0):
                            plt.ylabel('synchro', fontsize=10)
                        plt.imshow(output_MUSIC_ori, extent=(
                            np.amin(Loc_scan_x), np.amax(Loc_scan_x), np.amin(Loc_scan_y), np.amax(Loc_scan_y)),
                                   cmap=cm.jet)
                        localization_error = localization_error + np.sqrt(np.square(source_localization[bat, 0] - x_s)
                                                                          + np.square(2 - source_localization[bat, 1]
                                                                                      - y_s))

                        bat = bat + 1

                    plt.savefig(os.path.join(save_dir_fig, str(epoch * iters_epoch_train * batch_size + iters)))

                    plt.pause(5)
                    plt.close('all')
                    localization_error = localization_error / batch_size
                    print("mean localization_error is: ", localization_error)
                    localization_error = 0








