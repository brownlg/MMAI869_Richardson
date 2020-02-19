# https://adventuresinmachinelearning.com/python-tensorflow-tutorial/
# Python optimisation variables

import tensorflow as tf

learning_rate = 0.5
epochs = 10
batch_size = 100

from tensorflow.training.MNIST_data import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# declare the training data placeholders
# input x - for 28 x 28 pixels = 784
x = tf.placeholder(tf.float32, [None, 784])
# now declare the output data placeholder - 10 digits
y = tf.placeholder(tf.float32, [None, 10])