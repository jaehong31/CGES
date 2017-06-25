import tensorflow as tf


def weight_variable(shape, name_idx):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, name='weight' + str(name_idx))

def bias_variable(shape, name_idx):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, name='bias' + str(name_idx))

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def mnist_conv(x, num_classes, keep_prob):
  
  filter1 = weight_variable([5, 5, 1, 32], 1)
  bias1 = bias_variable([32], 1)
  
  x_ = tf.reshape(x, [-1, 28, 28, 1])
  
  relu1 = tf.nn.relu(conv2d(x_, filter1) + bias1)
  pool1 = max_pool_2x2(relu1)
    
  filter2 = weight_variable([5, 5, 32, 64], 2)
  bias2 = bias_variable([64], 2)

  relu2 = tf.nn.relu(conv2d(pool1, filter2) + bias2)
  pool2 = max_pool_2x2(relu2)

  fc3 = weight_variable([1024, 7 * 7 * 64], 3)
  bias3 = bias_variable([1024], 3)
#  fc3 = weight_variable([7 * 7 * 64, 1024], 3)
#  bias3 = bias_variable([1024], 3)

  flat_pool2 = tf.reshape(pool2, [-1, 7 * 7 * 64])
#  relu3 = tf.nn.relu(tf.matmul(flat_pool2, fc3) + bias3)
  relu3 = tf.nn.relu(tf.matmul(flat_pool2, tf.transpose(fc3)) + bias3)

  drop = tf.nn.dropout(relu3, keep_prob)
#  drop = relu3
  
#  fc4 = weight_variable([1024, 10], 4)
#  bias4 = bias_variable([10], 4)
  fc4 = weight_variable([10, 1024], 4)
  bias4 = bias_variable([10], 4)
#  output = tf.matmul(drop, fc4) + bias4
  output = tf.matmul(drop, tf.transpose(fc4)) + bias4

  return tf.nn.softmax(output)

  
  

