# coding: utf-8

# Deep Learning
# =============
# 任务2
# Assignment 2
# ------------
# 之前在1_notmnist.ipynb中，创建了一个封存格式化的数据集用来训练、开发和测试
# Previously in `1_notmnist.ipynb`, we created a pickle with formatted datasets for training, development and testing on the [notMNIST dataset](http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html).
# 这个任务的目标是逐步训练更深更精确的模型使用TensorFlow
# The goal of this assignment is to progressively train deeper and more accurate models using TensorFlow.

# In[ ]:

# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range

# First reload the data we generated in `1_notmnist.ipynb`.

# In[ ]:

pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']
    del save  # hint to help gc free up memory
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)

# 重新格式化数据让模型更适合训练，数据为扁平矩阵，标签为浮点的1位有效编码
# Reformat into a shape that's more adapted to the models we're going to train:
# - data as a flat matrix,
# - labels as float 1-hot encodings.
# 
# In[ ]:
# 图片尺寸和分类数量
image_size = 28
num_labels = 10


def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return dataset, labels

# 重新格式化数据集、格式化逻辑：TODO
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


# 我们首先训练一个多项式逻辑回归模型使用简单梯度下降算法
# We're first going to train a multinomial logistic regression using simple gradient(梯度) descent(下降).
#
# TensorFlow works like this:
#   首先描述要执行的计算：定义输入、变量和操作。这些作为计算图上的节点创建。此描述全部包含在下面的块中
# * First you describe the computation that you want to see performed: what the inputs, the variables, and the operations look like. These get created as nodes over a computation graph. This description is all contained within the block below:
#
#       with graph.as_default():
#           ...
#   然后你可以使用session.run来运行图上的操作，想几次就几次，输出从返回的图中获取。运行时操作包含在下面的块中
# * Then you can run the operations on this graph as many times as you want by calling `session.run()`, providing it outputs to fetch from the graph that get returned. This runtime operation is all contained in the block below:
#
#       with tf.Session(graph=graph) as session:
#           ...
# 让我们加载全部数据到TensorFlow中，建立与训练相对应的计算图
# Let's load all the data into TensorFlow and build the computation graph corresponding to our training:

# In[ ]:
# 随着梯度下降训练，即使如此多的数据是禁止的，训练数据子集以更快的周转
# With gradient descent training, even this much data is prohibitive.
# Subset the training data for faster turnaround.
train_subset = 10000

graph = tf.Graph()
with graph.as_default():
    # Input data.
    # 加载训练、验证和测试数据连接到图上的常量中
    # Load the training, validation and test data into constants that are attached to the graph.
    tf_train_dataset = tf.constant(train_dataset[:train_subset, :])
    tf_train_labels = tf.constant(train_labels[:train_subset])
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    # Variables.
    # 这些是我们将训练的参数，权重矩阵将使用一个（截断）正态分布的随机值初始化。偏差被初始化为0
    # These are the parameters that we are going to be training. The weight matrix will be initialized using random values following a (truncated) normal distribution. The biases get initialized to zero.
    weights = tf.Variable(tf.truncated_normal([image_size * image_size, num_labels]))
    biases = tf.Variable(tf.zeros([num_labels]))

    # 训练计算
    # Training computation. 
    # 我们将输入与权重矩阵相乘并加上偏差，然后计算softmax和交叉熵（cross-entropy）
    # We multiply the inputs with the weight matrix, and add biases. We compute the softmax and cross-entropy 
    # 这在TensorFlow中是一个操作，因为比较常见且最优化
    # (it's one operation in TensorFlow, because it's very common, and it can be optimized). 
    # 我们取所有训练样本交叉熵的平均值，这就是我们的损失值
    # We take the average of this cross-entropy across all training examples: that's our loss.
    logits = tf.matmul(tf_train_dataset, weights) + biases
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))

    # 优化器
    # Optimizer.
    # 使用梯度下降法来寻找这个损失最小值
    # We are going to find the minimum of this loss using gradient descent.
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    # 预测训练集、验证集和测试集
    # Predictions for the training, validation, and test data.
    # 这些不是训练的一部分，在这里我们可以在训练时报告准确数字
    # These are not part of training, but merely here so that we can report accuracy figures as we train.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(tf.matmul(tf_valid_dataset, weights) + biases)
    test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)

# Let's run this computation and iterate:

# In[ ]:

num_steps = 801


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])


with tf.Session(graph=graph) as session:
    # 这是个一次性操作，保证了参数的初始化，正如我们在图中描述的：矩阵的随机权重，0偏差
    # This is a one-time operation which ensures the parameters get initialized as
    # we described in the graph: random weights for the matrix, zeros for the biases.
    tf.initialize_all_variables().run()
    print('Initialized')
    for step in range(num_steps):
        # 运行计算，我们告诉.run()我们想运行优化器并获得损失值和训练预测返回的numpy数组
        # Run the computations. We tell .run() that we want to run the optimizer,
        # and get the loss value and the training predictions returned as numpy arrays.
        _, l, predictions = session.run([optimizer, loss, train_prediction])
        if (step % 100 == 0):
            print('Loss at step %d: %f' % (step, l))
            print('Training accuracy: %.1f%%' % accuracy(predictions, train_labels[:train_subset, :]))
            # 调用.eval()在验证预测上像调用.run()，但只为了得到一个numpy数组。注意所有的图重新依赖
            # Calling .eval() on valid_prediction is basically like calling run(), but
            # just to get that one numpy array. Note that it recomputes all its graph dependencies.
            print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_labels))
    print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))

# 现在让我们使用随机梯度下降法训练来替代，看哪个更快
# Let's now switch to stochastic gradient descent training instead, which is much faster.
# 图看起来相似，除了将所有数据保存到一个常量节点中，我们创建一个“占位符”节点，它将在session.run()的每次调用中输入实际数据
# The graph will be similar, except that instead of holding all the training data into a constant node, we create a `Placeholder` node which will be fed actual data at every call of `session.run()`.

# In[ ]:

batch_size = 128

graph = tf.Graph()
with graph.as_default():
    # 输入数据，对于训练数据，我们用一个占位符，运行时小批量的喂训练数据，
    # Input data. For the training data, we use a placeholder that will be fed at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.float32,shape=(batch_size, image_size * image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    # Variables.
    weights = tf.Variable(tf.truncated_normal([image_size * image_size, num_labels]))
    biases = tf.Variable(tf.zeros([num_labels]))

    # Training computation.
    logits = tf.matmul(tf_train_dataset, weights) + biases
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))

    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(tf.matmul(tf_valid_dataset, weights) + biases)
    test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)

# Let's run it:

# In[ ]:

num_steps = 3001

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print("Initialized")
    for step in range(num_steps):
        # 在训练数据中选择一个偏移量，该数据已被随机化。注意：我们可以使用更好的跨时代随机化
        # Pick an offset within the training data, which has been randomized.
        # Note: we could use better randomization across epochs.
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        # Generate a minibatch.生成小批次
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        # 准备一个字典（映射）告诉session在哪喂小批量数据，字典的key是图的占位符节点需要喂数据的，字典的值是numpy数组，用来喂节点
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 500 == 0):
            print("Minibatch loss at step %d: %f" % (step, l))
            print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
            print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))
    print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))

# ---
# Problem
# -------
# 改变逻辑回归例子，使用SGD（随机梯度下降）拥有一个隐藏层的神经网络，该神经网络有调整的线性单元和1024个隐藏节点。该模型应该提高你的验证/测试准确率
# Turn the logistic regression example with SGD into a 1-hidden layer neural network
# with rectified linear units [nn.relu()](https://www.tensorflow.org/versions/r0.7/api_docs/python/nn.html#relu)
# and 1024 hidden nodes. This model should improve your validation / test accuracy.
#
# ---

batch_size = 128
hidden_size = 1024

graph = tf.Graph()
with graph.as_default():
    # 输入数据，对于训练数据来说，使用一个将要训练的占位符在运行时训练一小批
    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.float32,shape=(batch_size, image_size * image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    # Variables.
    W1 = tf.Variable(tf.truncated_normal([image_size * image_size, hidden_size]))
    b1 = tf.Variable(tf.zeros([hidden_size]))

    W2 = tf.Variable(tf.truncated_normal([hidden_size, num_labels]))
    b2 = tf.Variable(tf.zeros([num_labels]))

    # Training computation.
    y1 = tf.nn.relu(tf.matmul(tf_train_dataset, W1) + b1)
    logits = tf.matmul(y1, W2) + b2

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))

    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)

    y1_valid = tf.nn.relu(tf.matmul(tf_valid_dataset, W1) + b1)
    valid_logits = tf.matmul(y1_valid, W2) + b2
    valid_prediction = tf.nn.softmax(valid_logits)

    y1_test = tf.nn.relu(tf.matmul(tf_test_dataset, W1) + b1)
    test_logits = tf.matmul(y1_test, W2) + b2
    test_prediction = tf.nn.softmax(test_logits)

# Let's run it:
num_steps = 3001

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print("Initialized")
    for step in range(num_steps):
        # Pick an offset within the training data, which has been randomized.
        # Note: we could use better randomization across epochs.
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        # Generate a minibatch.
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 500 == 0):
            print("Minibatch loss at step %d: %f" % (step, l))
            print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
            print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))
    print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
