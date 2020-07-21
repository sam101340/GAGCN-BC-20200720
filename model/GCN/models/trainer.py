import time
from os.path import join as pjoin
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from model.GCN.models.base_model import build_model, model_save
from model.GCN.models.tester import model_inference
from model.GCN.data_loader.data_utils import gen_batch

from model.GAT.main_gat import *


def model_train(inputs, blocks, args, sum_path='./output/tensorboard'):

    n, n_his, n_pred = args.n_route, args.n_his, args.n_pred
    Ks, Kt = args.ks, args.kt
    batch_size, epoch, inf_mode, opt = args.batch_size, args.epoch, args.inf_mode, args.opt

    # Placeholder for model training
    x = tf.placeholder(tf.float32, [None, n_his + 1, n, 1], name='data_input')
    traffic_data_tf = tf.placeholder(tf.float32, [None, n, 3], name='Gat_data_input')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    #get Gat
    #dis_data = creat_distance.norm_dis(dis_path)
    W_ma = get_weighted_adj_matrix(traffic_data_tf,n)
    Lk = first_approx(W_ma,n)
    #Lk = cheb_poly_approx(L,Ks,n)
    tf.add_to_collection(name='graph_kernel', value=tf.cast(Lk, tf.float32))

    # Define model loss
    train_loss, pred = build_model(x, n_his, Ks, Kt, blocks, keep_prob)
    tf.summary.scalar('train_loss', train_loss)
    copy_loss = tf.add_n(tf.get_collection('copy_loss'))
    tf.summary.scalar('copy_loss', copy_loss)

    # Learning rate settings
    global_steps = tf.Variable(0, trainable=False)
    len_train = inputs.get_len('train')
    if len_train % batch_size == 0:
        epoch_step = len_train / batch_size
    else:
        epoch_step = int(len_train / batch_size) + 1
    # Learning rate decay with rate 0.7 every 5 epochs.
    lr = tf.train.exponential_decay(args.lr, global_steps, decay_steps=5 * epoch_step, decay_rate=0.7, staircase=True)
    tf.summary.scalar('learning_rate', lr)
    step_op = tf.assign_add(global_steps, 1)
    with tf.control_dependencies([step_op]):
        if opt == 'RMSProp':
            train_op = tf.train.RMSPropOptimizer(lr).minimize(train_loss)
        elif opt == 'ADAM':
            train_op = tf.train.AdamOptimizer(lr).minimize(train_loss)
        else:
            raise ValueError(f'ERROR: optimizer "{opt}" is not defined.')

    merged = tf.summary.merge_all()

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        writer = tf.summary.FileWriter(pjoin(sum_path, 'train'), sess.graph)
        sess.run(tf.global_variables_initializer())

        if inf_mode == 'sep':
            # for inference mode 'sep', the type of step index is int.
            step_idx = n_pred - 1
            tmp_idx = [step_idx]
            min_val = min_va_val = np.array([4e1, 1e5, 1e5])
        elif inf_mode == 'merge':
            # for inference mode 'merge', the type of step index is np.ndarray.
            step_idx = tmp_idx = np.arange(3, n_pred + 1, 3) - 1
            min_val = min_va_val = np.array([4e1, 1e5, 1e5] * len(step_idx))
        else:
            raise ValueError(f'ERROR: test mode "{inf_mode}" is not defined.')

        for i in range(epoch):
            start_time = time.time()
            #Gat_start_idx = 0
            for j, x_batch in enumerate(
                    gen_batch(inputs.get_data('train'), batch_size, dynamic_batch=True, shuffle=True)):
                #Gat_end_idx = Gat_start_idx + batch_size
                summary, _ = sess.run([merged, train_op], feed_dict={x: x_batch[:, 0:n_his + 1, :, 2:3], keep_prob: 1.0,traffic_data_tf:x_batch.reshape(-1,n,3)})
                #Gat_start_idx = Gat_end_idx
                writer.add_summary(summary, i * epoch_step + j)
                if j % 50 == 0:
                    loss_value = \
                        sess.run([train_loss, copy_loss],
                                 feed_dict={x: x_batch[:, 0:n_his + 1, :,2:3], keep_prob: 1.0,traffic_data_tf:x_batch.reshape(-1,n,3)})
                    print(f'Epoch {i:2d}, Step {j:3d}: [{loss_value[0]:.3f}, {loss_value[1]:.3f}]')
            print(f'Epoch {i:2d} Training Time {time.time() - start_time:.3f}s')

            start_time = time.time()
            min_va_val, min_val = \
                model_inference(sess, pred, inputs, batch_size, n_his, n_pred, step_idx, min_va_val, min_val)

            for ix in tmp_idx:
                va, te = min_va_val[ix - 2:ix + 1], min_val[ix - 2:ix + 1]
                print(f'Time Step {ix + 1}: '
                      f'MAPE {va[0]:7.3%}, {te[0]:7.3%}; '
                      f'MAE  {va[1]:4.3f}, {te[1]:4.3f}; '
                      f'RMSE {va[2]:6.3f}, {te[2]:6.3f}.')
            print(f'Epoch {i:2d} Inference Time {time.time() - start_time:.3f}s')

            if (i + 1) % args.save == 0:
                model_save(sess, global_steps, 'GCN')
        writer.close()
    print('Training model finished!')

