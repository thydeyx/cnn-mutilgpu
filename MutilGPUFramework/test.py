# -*- coding:utf-8 -*-
#
#        Author : TangHanYi
#        E-mail : thydeyx@163.com
#   Create Date : 2017-09-10 01:04:07
# Last modified : 2017-09-12 16时42分21秒
#     File Name : test.py
#          Desc :

import sys
sys.path.append('../')
from MutilGPUFramework.framework import MutilGPUFrameWork
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

class Solution(MutilGPUFrameWork):

    def __init__(self, gpu_num=4):
        super(Solution, self).__init__(gpu_num=gpu_num)

    def get_weight_varible(self, name, shape):
        return tf.get_variable(name, shape=shape,
                               initializer=tf.contrib.layers.xavier_initializer())

    def get_bias_varible(self, name, shape):
        return tf.get_variable(name, shape=shape,
                               initializer=tf.contrib.layers.xavier_initializer())

    # filter_shape: [f_h, f_w, f_ic, f_oc]
    def conv2d(self, layer_name, x, filter_shape):
        with tf.variable_scope(layer_name):
            w = self.get_weight_varible('w', filter_shape)
            b = self.get_bias_varible('b', filter_shape[-1])
            y = tf.nn.bias_add(tf.nn.conv2d(input=x, filter=w, strides=[1, 1, 1, 1], padding='SAME'), b)
            return y

    def pool2d(self, layer_name, x):
        with tf.variable_scope(layer_name):
            y = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            return y

    # inp_shape: [N, L]
    # out_shape: [N, L]
    def fc(self, layer_name, x, inp_shape, out_shape):
        with tf.variable_scope(layer_name):
            inp_dim = inp_shape[-1]
            out_dim = out_shape[-1]
            y = tf.reshape(x, shape=inp_shape)
            w = self.get_weight_varible('w', [inp_dim, out_dim])
            b = self.get_bias_varible('b', [out_dim])
            y = tf.add(tf.matmul(y, w), b)
            return y

    def build_model(self, x):
        y = tf.reshape(x,shape=[-1, 28, 28, 1])
        #layer 1
        y = self.conv2d('conv_1', y, [3, 3, 1, 8])
        y = self.pool2d('pool_1', y)
        #layer 2
        y = self.conv2d('conv_2', y, [3, 3, 8, 16])
        y = self.pool2d('pool_2', y)
        #layer fc
        y = self.fc('fc', y, [-1, 7*7*16], [-1, 10])
        return y

    def read_data(self):
        mnist = input_data.read_data_sets('/tmp/data/mnist', one_hot=True)
        return mnist

    def config_para(self):
        self.batch_size = 128 * self.gpu_num
        self.func = self.build_model
        data = self.read_data()
        self.train_datas = np.array(data.train.images)
        self.train_labels = np.array(data.train.labels)
        self.test_datas = np.array(data.test.images)
        self.test_labels = np.array(data.test.labels)
        #config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.session_config = config
        self.lr = 1e-2
        self.opt = 'Adam'
        self.epochs = 2000

    def train(self):
        self.config_para()
        self.run()

if __name__ == "__main__":
    s = Solution()
    s.train()
