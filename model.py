# -*- coding:utf-8 -*-
#
#        Author : TangHanYi
#        E-mail : thydeyx@163.com
#   Create Date : 2017-09-07 18时42分01秒
# Last modified : 2017-09-07 18时42分05秒
#     File Name : model.py
#          Desc :

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import load_dataset

def cnn_model_fn(features, labels, mode):
    input_layer = tf.reshape(features['x'], [-1,28,28,1])
    conv1 = tf.layers.conv2d(inputs=input_layer,
                             filters=32,
                             kernel_size=[5, 5],
                             padding='same',
                             activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5,5],
        padding='same',
        activation=tf.nn.relu
    )
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    pool2_flat = tf.reshape(pool2, [-1, 7*7*64])
    dense = tf.layers.dense(pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
    logits = tf.layers.dense(dropout, units=10)
    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    onehot_labels = tf.one_hot(indices=tf.cast(labels, dtype=tf.int32), depth=10)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss,
                                      global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"]
        )
    }

    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops
    )

def main(_):
    mnist = load_dataset('mnist')
    train_data = mnist.train.images
    train_labels = mnist.train.labels
    test_data = mnist.test.images
    test_labels = mnist.test.labels
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir='./model/'
    )
    #tf.estimator.EstimatorSpec()
    tensors_to_log = {"probabilities":"softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True
    )
    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=20000,
        hooks=[logging_hook]
    )
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x':test_data},
        y=test_labels,
        num_epochs=1,
        shuffle=False
    )
    eval_result = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_result)

if __name__ == "__main__":
    #print(mode == tf.estimator.ModeKeys.TRAIN)
    tf.app.run()
