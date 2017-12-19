#! usr/bin/python
#coding=utf-8
'''
Created on 2017年6月27日

@author: lianglong
'''
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from test_tensorflow.model.logistic_regression import correct_prediction

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# keep_prob = tf.placeholder(dtype=tf.float32)

def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
#     Wx_plus_b = tf.nn.dropout(Wx_plus_b)
    if activation_function is None:
        outputs = Wx_plus_b
    else :
        outputs = activation_function(Wx_plus_b)
    return outputs


def compute_accuracy(v_xs, v_ys):
    global prediction
    # y_pre输出结果为矩阵，每一行的数据代表各种分类的预测概率，每列代表一个分类
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    # 拿预测结果和标签结果进行对比，每一行取最大值，得到每一类的预测
    correct_prediction = tf.equal(tf.arg_max(y_pre, 1), tf.arg_max(v_ys, 1))
    # 计算一组数据预测的平均值，cast转换类型
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # 运行结果
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys:v_ys})
    return result

xs = tf.placeholder(dtype=tf.float32, shape=[None, 784])
ys = tf.placeholder(dtype=tf.float32, shape=[None, 10])


prediction = add_layer(xs, 784, 10, activation_function=tf.nn.softmax)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

for i in range(10000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={xs: batch_xs, ys:batch_ys})
    if i%50 == 0:
        print(compute_accuracy(mnist.test.images, mnist.test.labels))

