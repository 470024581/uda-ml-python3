# coding: utf-8
'''
A Convolutional Network implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("tmp/data/", one_hot=True)


# Parameters 学习率 训练循环次数 每份大小 显示步长
learning_rate = 0.001
training_iters = 100000
# training_iters = 200000
batch_size = 128
display_step = 10

# Network Parameters
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units 保持单位概率（丢弃率=1-dropout=0.25）

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

# Create some wrappers for simplicity  训练数据、权重、偏置、步长
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    # 参数一input：需要卷积的输入图像、要求是一个4维类型为float32或float64的Tensor具有[batch, in_height, in_wight, in_channels]的shape分别是[单批训练样本数，图片高度，图片宽度，输入通道数（深度）]
    # 参数二filter：卷积核（权重/过滤器）、要求类型与参数1相同，通道数与参数1通道数一致，[filter_height, filter_width, in_channels, out_channels]的shape分别是[卷积核高度，卷积核宽度，输入通道数，输出通道数（卷积核个数）]
    # 参数三strides：卷积时在图像每一维的步长，类型是一维向量长度4，strides[0]=strides[3]=1
    # 参数四padding：补白方式SAME或者VALID，SMAE以0补白，VALID为有效补白
    # 参数五use_cudnn_on_gpu：bool类型，是否使用cudnn加速，默认为true
    # 返回值：Tensor类型的feature map
    x = tf.nn.conv2d(input=x, filter=W, strides=[1, strides, strides, 1], padding='SAME', use_cudnn_on_gpu=None)
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x, name="RELU") # name 可以指定Tensor的名字


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    # 参数1：4D张量，[batch, height, width, channels] 与conv2d的input格式一致
    # 参数2：长为4的list，表示池化窗口的尺寸
    # 参数3：池化窗口的滑动值，与conv2d中用法一致
    # 参数4：与conv2d中用法一致
    return tf.nn.max_pool(value=x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    print(x) # Tensor("Placeholder:0", shape=(?, 784), dtype=float32) 
    x = tf.reshape(x, shape=[-1, 28, 28, 1]) # 
    print(x) # Tensor("Reshape:0", shape=(?, 28, 28, 1), dtype=float32)
    
    # 卷积核尺寸为5*5，输入通道为1，输出通道32（feature map数量为32），因为步长为1所以输出尺寸和输入图像一样，单个通道的输出为28*28，有？个批次，32个通道，所以结果如下
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    print(conv1) # Tensor("Relu:0", shape=(?, 28, 28, 32), dtype=float32)
    
    # 最大池化（下采样），ksize=[1,2,2,1]，那么卷积结果(?*28*28*32)经过池化后为?*14*14*32
    conv1 = maxpool2d(conv1, k=2)
    print(conv1) # Tensor("MaxPool:0", shape=(?, 14, 14, 32), dtype=float32)

    # 卷积前尺寸为(?*14*14*32)，卷积后为?*14*14*64
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # 池化后为?*7*7*64
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer 全连接层
    # Reshape conv2 output to fit fully connected layer input 重构conv2层的输出，来适应全连接层的输入
    # reshape结构为7*7*64
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    print(fc1) # Tensor("Reshape_1:0", shape=(?, 3136), dtype=float32)
    # 输入为[?, 7*7*64]，乘以权重矩阵[7*7*64, 1024]，输出通道为[?, 1024]
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    print(fc1) # Tensor("Add:0", shape=(?, 1024), dtype=float32)
    fc1 = tf.nn.relu(fc1)
    print(fc1) # Tensor("Relu:0", shape=(?, 1024), dtype=float32)
    fc1 = tf.nn.dropout(fc1, dropout)
    print(fc1) # Tensor("dropout/mul:0", shape=(?, 1024), dtype=float32)
    # Output, class prediction
    # 输入为[?, 1024]，乘以权重矩阵[1024, 10]，输出通道为[?, 10]
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    print(out)
    return out

# Store layers weight & bias 
weights = {
    # 5x5 conv, 1 input, 32 outputs 卷积核尺寸5*5，输入通道1，输出通道32
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs（depth）, 64 outputs（depth）输入通道32，输出通道64
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer 使用softmax_cross_entropy_with_logits激活函数，计算损失值，定义AdamOptimizer优化器
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model 评估模型
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations 一直训练直到最大迭代
    while step * batch_size < training_iters:
        # 获取batch_size128个训练样本和标记
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop) 运行优化器操作（反向传播）
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
        if step % display_step == 0:
            # Calculate batch loss and accuracy 计算每一批次的损失值和准确率
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    # Calculate accuracy for 256 mnist test images 计算测试集的准确率
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: mnist.test.images[:256], y: mnist.test.labels[:256], keep_prob: 1.}))
