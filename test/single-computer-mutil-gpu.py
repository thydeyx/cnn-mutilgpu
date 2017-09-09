# -*- coding:utf-8 -*-
import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import load_dataset

def get_weight_variable(name, shape):
    return tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())

def get_bias_variable(name, shape):
    return tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())

def conv2d(layer_name, x, filter_shape):
    with tf.variable_scope(layer_name):
        w = tf.get_variable('w', filter_shape)
        b = tf.get_variable('b', filter_shape[-1])
        y = tf.nn.bias_add(tf.nn.conv2d(input=x, filter=w, strides=[1,1,1,1], padding='SAME'), b)
        return y

def pool2d(layer_name, x):
    with tf.variable_scope(layer_name):
        y = tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        return y

def fc(layer_name, x, input_shape, out_shape):
    with tf.variable_scope(layer_name):
        inp_dim = input_shape[-1]
        out_dim = out_shape[-1]
        x = tf.reshape(x, shape=input_shape)
        w = tf.get_variable('w', [inp_dim, out_dim])
        b = tf.get_variable('b', out_dim)
        y = tf.add(tf.matmul(x, w), b)
        return y

def build_model(x):
    y = tf.reshape(x, shape=[-1,28,28,1])
    y = conv2d('conv_1', y, [3,3,1,32])
    y = pool2d('pool_1', y)

    y = conv2d('conv_2', y, [3,3,32,64])
    y = pool2d('pool_2', y)

    y = fc('fc', y, [-1, 7*7*64], [-1, 10])
    return y

def one_hot(x):
    n = len(x)
    batch_y = np.zeros((n, 10), dtype=np.int32)
    for i in range(n):
        batch_y[i][x[i]] = 1
    return batch_y

def single_gpu():
    batch_size = 128
    epochs = 10
    mnist = load_dataset('mnist')
    #train_data = mnist.train.images
    #train_labels = mnist.train.labels
    test_data = mnist.test.images
    test_labels = mnist.test.labels

    #train_labels = tf.one_hot(indices=tf.cast(train_labels, dtype=tf.int32), depth=10)
    #test_labels = tf.one_hot(indices=tf.cast(test_labels, dtype=tf.int32), depth=10)
    tf.reset_default_graph()
    config = tf.ConfigProto(log_device_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        print('build model')
        x = tf.placeholder(tf.float32, [None, 784])
        y = tf.placeholder(tf.float32, [None, 10])
        pred = build_model(x)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
        learning_rate = tf.placeholder(tf.float32, shape=[])
        train_op = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(loss)
        all_y = tf.reshape(y, [-1, 10])
        all_pred = tf.reshape(pred, [-1, 10])
        correct_pred = tf.equal(tf.argmax(all_y, 1), tf.argmax(all_pred, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        print('run train op')
        sess.run(tf.global_variables_initializer())
        lr = 0.01
        for epoch in range(epochs):
            start_time = time.time()
            total_batch = int(mnist.train.num_examples/batch_size)
            avg_loss = 0.0
            print('\n---------------------')
            print('Epoch:%d, lr:%.4f' % (epoch, lr))
            for batch_idx in range(total_batch):
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                inp_dict = {}
                inp_dict[learning_rate] = lr
                inp_dict[x] = batch_x
                inp_dict[y] = one_hot(batch_y)
                _, _loss = sess.run([train_op, loss], inp_dict)
                avg_loss += _loss
            avg_loss /= total_batch
            print('Train loss:%.4f' % (avg_loss))
            lr = max(lr * 0.7, 0.00001)
            val_accuracy = sess.run([accuracy], {x: test_data, y:one_hot(test_labels)})[0]
            #print(val_accuracy)
            print('Val Accuracy: %0.4f%%' % (100.0 * val_accuracy))
        stop_time = time.time()
        elapsed_time = stop_time - start_time
        print('Cost time: ' + str(elapsed_time) + ' sec.')
    print('training done.')

def averge_losses(loss):
    tf.add_to_collection()

if __name__ == '__main__':
    #print(get_weight__variable('heh', [2,2]))
    single_gpu()
