# -*- coding:utf-8 -*-
#
#        Author : TangHanYi
#        E-mail : thydeyx@163.com
#   Create Date : 2017-09-10 00:47:48
# Last modified : 2017-09-12 16时43分52秒
#     File Name : framework.py
#          Desc :

import tensorflow as tf
import numpy as np
import time

class MutilGPUFrameWork(object):

    def __init__(self, mode='tower', gpu_num = 4):
        self.mode = mode        #是否使用tower结构
        self.gpu_num = gpu_num  #GPU编号
        self.num_gpu = 0        #GPU数量
        self.batch_size = 128   #默认的每次每个GPU训练的数据集合大小
        self.train_datas = None  #训练集numpy
        self.train_labels = None #训练集标注
        self.test_datas = None   #测试集numpy
        self.test_labels = None  #测试集标注
        self.vaild_data = None  #验证集numpy
        self.func = None       #模型
        self.session_config = None #session配置
        self.lr = 1e-3          #模型学习率
        self.epochs = 10         #学习次数
        self.opt = None         #优化器设置
        self.train_shape = None
        self.test_shape = None
        self.train_data_num = None
        self.test_data_num = None
        self.train_label_shape = None
        self.test_label_shape = None
        self.test_label_num = None
        self.train_label_num = None
        #self.func = func

    def average_gradients(self, tower_grads):
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # 每一个grad_ans_vars保存每个GPU算出来的某个变量对应的梯度
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            # 每个GPU计算得到的变量vari和梯度集合
            grads = [g for g, _ in grad_and_vars]
            # Average over the 'tower' dimension.
            grad = tf.stack(grads, 0)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads

    def next_batch(self, datas, labels, idx, num):
        if (idx + 1) * self.batch_size > num:
            end = num
        else:
            end = (idx + 1) * self.batch_size
        start = idx * self.batch_size
        return datas[start:end], labels[start:end]

    def feed_all_gpu(self, inp_dict, models, payload_per_gpu, batch_x, batch_y):
        for i in range(len(models)):
            x, y, _, _, _ = models[i]
            start_pos = i * payload_per_gpu
            stop_pos = (i + 1) * payload_per_gpu
            inp_dict[x] = batch_x[start_pos:stop_pos]
            inp_dict[y] = batch_y[start_pos:stop_pos]
        return inp_dict

    def train_tower(self):

        tf.reset_default_graph()
        if self.session_config is None:
            self.session_config = tf.ConfigProto()
        with tf.Session(config=self.session_config) as sess:
            with tf.device('/cpu:0'):
                learning_rate = tf.placeholder(tf.float32, shape=[])
                opt = None
                if self.opt == 'Adam':
                    opt = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)
                elif self.opt == 'SGD':
                    opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

                print('Build model on gpu tower...')
                models = []
                not_first_gpu = False
                for gpu_id in self.gpu_num:
                    with tf.device('/gpu:%d' % gpu_id):
                        print('Tower:%d...' % gpu_id)
                        with tf.name_scope('tower_%id' % gpu_id):
                            with tf.variable_scope('cpu_variables', reuse=not_first_gpu):
                                not_first_gpu = True
                                #print(self.train_shape, self.train_label_shape)
                                x = tf.placeholder(tf.float32, self.train_shape)
                                y = tf.placeholder(tf.float32, self.train_label_shape)
                                pred = self.func(x)
                                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
                                grads = opt.compute_gradients(loss)
                                models.append((x, y, pred, loss, grads))
                print('Build model on gpu tower is complete.')

                print('Reduce model on cpu...')
                tower_x, tower_y, tower_preds, tower_losses, tower_grads = zip(*models)
                aver_loss_op = tf.reduce_mean(tower_losses)
                apply_gradient_op = opt.apply_gradients(self.average_gradients(tower_grads))

                all_y = tf.reshape(tf.stack(tower_y, 0), [-1, self.train_label_shape[1]])
                all_pred = tf.reshape(tf.stack(tower_preds, 0), [-1, self.train_label_shape[1]])
                correct_pred = tf.equal(tf.argmax(all_y, 1), tf.argmax(all_pred, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_pred, 'float'))
                print('reduce model on cpu done.')

                print('run train op...')
                # 初始化
                sess.run(tf.global_variables_initializer())
                lr = self.lr
                for epoch in range(self.epochs):
                    start_time = time.time()
                    # 每个GPU的数据量
                    payload_per_gpu = int(self.batch_size / self.num_gpu)
                    total_batch = int(self.train_data_num / self.batch_size)
                    avg_loss = 0.0
                    print('\n---------------------')
                    print('Epoch:%d, lr:%.4f' % (epoch, lr))
                    for batch_idx in range(total_batch):
                        batch_x, batch_y = self.next_batch(self.train_datas, self.train_labels, batch_idx, self.train_data_num)
                        inp_dict = {}
                        inp_dict[learning_rate] = lr
                        inp_dict = self.feed_all_gpu(inp_dict, models, payload_per_gpu, batch_x, batch_y)
                        _, _loss = sess.run([apply_gradient_op, aver_loss_op], inp_dict)
                        avg_loss += _loss
                    avg_loss /= total_batch
                    print('Train loss:%.4f' % (avg_loss))

                    #lr = max(lr * 0.7, 0.00001)

                    val_payload_per_gpu = int(self.batch_size / self.num_gpu)
                    total_batch = int(self.test_label_num / self.batch_size)
                    preds = None
                    ys = None
                    for batch_idx in range(total_batch):
                        batch_x, batch_y = self.next_batch(self.test_datas, self.test_labels, batch_idx, self.test_data_num)
                        inp_dict = self.feed_all_gpu({}, models, val_payload_per_gpu, batch_x, batch_y)
                        batch_pred, batch_y = sess.run([all_pred, all_y], inp_dict)
                        if preds is None:
                            preds = batch_pred
                        else:
                            preds = np.concatenate((preds, batch_pred), 0)
                        if ys is None:
                            ys = batch_y
                        else:
                            ys = np.concatenate((ys, batch_y), 0)
                    val_accuracy = sess.run([accuracy], {all_y: ys, all_pred: preds})[0]
                    print('Val Accuracy: %0.4f%%' % (100.0 * val_accuracy))

                    stop_time = time.time()
                    elapsed_time = stop_time - start_time
                    print('Cost time: ' + str(elapsed_time) + ' sec.')
                print('training done.')

    def run(self):

        if self.func is None:
            print('Please specify the mode!')
            return

        if self.train_datas is None or self.test_datas is None or self.train_labels is None or self.test_labels is None:
            print('Do not find train and test data!')
            return

        if self.opt is None:
            print('Please specify the Optimizer!')
            return

        if type(self.gpu_num) == int:
            tmp = []
            for i in range(self.gpu_num):
                tmp.append(i)
            self.gpu_num = tmp[:]
        elif type(self.gpu_num) == list:
            for i in self.gpu_num:
                if type(i) != int:
                    print('In GPU_NUM list is int!')
                    return
        else:
            print('GPU_NUM is a int or list!')
            return
        self.num_gpu = len(self.gpu_num)

        if self.mode == 'tower':
            self.train_shape = self.train_datas.shape[1:]
            self.test_shape = self.test_datas.shape[1:]
            self.test_data_num = self.test_datas.shape[0]
            self.train_data_num = self.train_datas.shape[0]
            self.train_label_shape = self.train_labels.shape[1:]
            self.test_label_shape = self.test_labels.shape[1:]
            self.train_label_num = self.train_labels.shape[0]
            self.test_label_num = self.test_labels.shape[0]
            if self.train_shape != self.test_shape:
                print('The train shape ', self.train_shape, ' is not equal to test shape ', self.test_shape)
                return
            if self.train_label_shape != self.test_label_shape:
                print('The train label shape ', self.train_shape, ' is not equal to test label shape ', self.test_shape)
                return
            if self.train_data_num != self.train_label_num:
                print('The number of train data is not equal the number of train label.')
                return
            if self.test_data_num != self.test_label_num:
                print('The number of test data is not equal the number of test label')
                return
            print(self.train_label_shape, self.test_label_shape, list(self.train_label_shape))
            tmp = [None]
            for shape in self.train_shape:
                tmp.append(shape)
            self.train_shape = tmp[:]
            tmp = [None]
            for shape in self.train_label_shape:
                tmp.append(shape)
            self.train_label_shape = tmp[:]
            self.train_tower()

if __name__ == "__main__":
    pass
