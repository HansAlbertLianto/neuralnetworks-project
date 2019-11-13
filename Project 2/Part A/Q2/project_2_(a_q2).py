# -*- coding: utf-8 -*-
"""Project 2 (A-Q2).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1LMht7W5TQw1OKLPVvHrAwVxMGEZE-rEh
"""

# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 1.15
import tensorflow as tf
import numpy as np
import pylab as plt
import pickle

NUM_CLASSES = 10
IMG_SIZE = 32
NUM_CHANNELS = 3
learning_rate = 0.001
epochs = 800
batch_size = 128

seed = 10
np.random.seed(seed)
tf.set_random_seed(seed)

hyper_parameter = [(50,50), (50,70), (50,80), (50,90), (70,70), (70,80), 
                   (70,90), (80,80), (80, 90), (90,90)]

from tensorflow.python.client import device_lib
device_lib.list_local_devices()

def load_data(file):
  with open(file, 'rb') as fo:
      try:
          samples = pickle.load(fo)
      except UnicodeDecodeError:  # python 3.x
          fo.seek(0)
          samples = pickle.load(fo, encoding='latin1')

  data, labels = samples['data'], samples['labels']

  data = np.array(data, dtype=np.float32)
  labels = np.array(labels, dtype=np.int32)

  labels_ = np.zeros([labels.shape[0], NUM_CLASSES])
  labels_[np.arange(labels.shape[0]), labels - 1] = 1

  return data, labels_

def cnn_2(images, no_map1, no_map2):
  images = tf.reshape(images, [-1, IMG_SIZE, IMG_SIZE, NUM_CHANNELS])

  # Conv 1
  W_conv1 = tf.Variable(tf.truncated_normal([9, 9, NUM_CHANNELS, no_map1], stddev=1.0 / np.sqrt(NUM_CHANNELS * 9 * 9)),
                   name='weights_1')
  b_conv1 = tf.Variable(tf.zeros([no_map1]), name='biases_1')

  h_conv_1 = tf.nn.relu(tf.nn.conv2d(images, W_conv1, [1, 1, 1, 1], padding='VALID') + b_conv1)
  h_pool_1 = tf.nn.max_pool(h_conv_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool_1')

  # Conv 2
  W_conv2 = tf.Variable(tf.truncated_normal([5, 5, no_map1, no_map2], stddev=1.0 / np.sqrt(no_map1 * 5 * 5)),
                   name='weights_2')
  b_conv2 = tf.Variable(tf.zeros([no_map2]), name='biases_2')

  h_conv_2 = tf.nn.relu(tf.nn.conv2d(h_pool_1, W_conv2, [1, 1, 1, 1], padding='VALID') + b_conv2)
  h_pool_2 = tf.nn.max_pool(h_conv_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool_2')

  
  dim = h_pool_2.get_shape()[1].value * h_pool_2.get_shape()[2].value * h_pool_2.get_shape()[3].value
  # FC 1
  W_fc_1 = tf.Variable(tf.truncated_normal([dim, 300], stddev=1.0 / np.sqrt(dim)),
                   name='weights_3')
  b_fc_1 = tf.Variable(tf.zeros([300]), name='biases_3')
  
  pool_2_flat = tf.reshape(h_pool_2, [-1, dim])
  h_fc1 = tf.nn.relu(tf.matmul(pool_2_flat, W_fc_1) + b_fc_1)
  
  # Softmax
  W_fc_2 = tf.Variable(tf.truncated_normal([300, NUM_CLASSES], stddev=1.0 / np.sqrt(300)), name='weights_4')
  b_fc_2 = tf.Variable(tf.zeros([NUM_CLASSES]), name='biases_4')
  logits = tf.matmul(h_fc1, W_fc_2) + b_fc_2

  keep_prob = 1

  return W_conv1, h_conv_1, h_pool_1, h_conv_2, h_pool_2, logits, keep_prob

def train(no_map1, no_map2):  
  trainX, trainY = load_data('../data_batch_1')
  print(trainX.shape, trainY.shape)

  testX, testY = load_data('../test_batch_trim')
  print(testX.shape, testY.shape)

  trainX = (trainX - np.min(trainX, axis=0)) / np.max(trainX, axis=0)
  testX = (testX - np.min(trainX, axis=0)) / np.max(trainX, axis=0)
  with tf.device('/device:GPU:0'):

    # Create the model
    x = tf.placeholder(tf.float32, [None, IMG_SIZE * IMG_SIZE * NUM_CHANNELS])
    y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])

    W_conv1, h_conv_1, h_pool_1, h_conv_2, h_pool_2, logits, keep_prob = cnn_2(x, no_map1, no_map2)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=logits)
    loss = tf.reduce_mean(cross_entropy)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32) # Cast to float
    accuracy = tf.reduce_mean(correct_prediction)

    N = len(trainX)
    idx = np.arange(N)

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      test_acc = []
      train_cost = []

      for e in range(epochs):
        np.random.shuffle(idx)
        trainX, trainY = trainX[idx], trainY[idx]

        for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
          train_step.run(feed_dict={x: trainX[start:end], y_: trainY[start:end]})

        train_cost.append(loss.eval(feed_dict={x: trainX, y_: trainY}))
        test_acc.append(accuracy.eval(feed_dict={x: testX, y_: testY}))
        

        if e%100 == 0:
          print('Epoch {}: Train Cost = {}, Test Acc = {}'.format(e,train_cost[e], test_acc[e]))
          
  return train_cost, test_acc

hyper_parameter = [(10,30), (50,50), (80, 80)]
train_cost = []
test_acc = []
for h_para in hyper_parameter:
  train_result, test_result = train(h_para[0], h_para[1])
  train_cost.append(train_result)
  test_acc.append(test_result)

plt.figure(1)
for idx in range(len(train_cost)):
  plt.plot(range(epochs), train_cost[idx], label=str(hyper_parameter[idx][0])+" , " + str(hyper_parameter[idx][1]))
plt.xlabel('Number of Epochs')
plt.legend()
plt.ylabel('Training Cost')
plt.title("Training Cost Against Learning Epochs")
plt.show()
plt.savefig('./q2_train_cost_1-1.png')

plt.figure(1)
for idx in range(len(test_acc)):
  plt.plot(range(epochs), test_acc[idx], label=str(hyper_parameter[idx][0])+" , " + str(hyper_parameter[idx][1]))
plt.xlabel('Number of Epochs')
plt.legend()
plt.ylabel('Testing Accuracy')
plt.title("Testing Accuracy Against Learning Epochs")
plt.show()
plt.savefig('./q2_test_acc_1-1.png')

plt.figure(2)

paras = [testing[-1] for testing in test_acc]

plt.scatter(range(len(hyper_parameter)), paras)
plt.xticks(range(len(hyper_parameter)), hyper_parameter)


plt.xlabel('Number of Feature Maps')
plt.legend()
plt.ylabel('Testing Accuracy')
plt.title("Testing Accuracy Against Number of Feature Maps")
plt.show()
plt.savefig('./q2_test_acc_1-2.png')

hyper_parameter = [(50,80), (70,80), (80, 80),(80, 90), (85,90), (90,90)]
train_cost = []
test_acc = []
for h_para in hyper_parameter:
  train_result, test_result = train(h_para[0], h_para[1])
  train_cost.append(train_result)
  test_acc.append(test_result)

plt.figure(1)
for idx in range(len(train_cost)):
  plt.plot(range(epochs), train_cost[idx], label=str(hyper_parameter[idx][0])+" , " + str(hyper_parameter[idx][1]))
plt.xlabel('Number of Epochs')
plt.legend()
plt.ylabel('Training Cost')
plt.title("Training Cost Against Learning Epochs")
plt.show()
plt.savefig('./2_train_cost_2-1.png')

plt.figure(1)
for idx in range(len(test_acc)):
  plt.plot(range(epochs), test_acc[idx], label=str(hyper_parameter[idx][0])+" , " + str(hyper_parameter[idx][1]))
plt.xlabel('Number of Epochs')
plt.legend()
plt.ylabel('Testing Accuracy')
plt.title("Testing Accuracy Against Learning Epochs")
plt.show()
plt.savefig('./2_test_acc_2-1.png')

plt.figure(2)

paras = [testing[-1] for testing in test_acc]

plt.scatter(range(len(hyper_parameter)), paras)
plt.xticks(range(len(hyper_parameter)), hyper_parameter)

plt.xlabel('Number of Feature Maps')
plt.legend()
plt.ylabel('Testing Accuracy')
plt.title("Testing Accuracy Against Number of Feature Maps")
plt.show()
plt.savefig('./2_test_acc_2-2.png')

hyper_parameter = [(80,80), (80,85), (80, 90), (85,90), (90,90)]
train_cost = []
test_acc = []
for h_para in hyper_parameter:
  train_result, test_result = train(h_para[0], h_para[1])
  train_cost.append(train_result)
  test_acc.append(test_result)

plt.figure(1)
for idx in range(len(train_cost)):
  plt.plot(range(epochs), train_cost[idx], label=str(hyper_parameter[idx][0])+" , " + str(hyper_parameter[idx][1]))
plt.xlabel('Number of Epochs')
plt.legend()
plt.ylabel('Training Cost')
plt.title("Training Cost Against Learning Epochs")
plt.show()
plt.savefig('./2_train_cost_3-1.png')

plt.figure(1)
for idx in range(len(test_acc)):
  plt.plot(range(epochs), test_acc[idx], label=str(hyper_parameter[idx][0])+" , " + str(hyper_parameter[idx][1]))
plt.xlabel('Number of Epochs')
plt.legend()
plt.ylabel('Testing Accuracy')
plt.title("Testing Accuracy Against Learning Epochs")
plt.show()
plt.savefig('./2_test_acc_3-1.png')

plt.figure(2)

paras = [testing[-1] for testing in test_acc]

plt.scatter(range(len(hyper_parameter)), paras)
plt.xticks(range(len(hyper_parameter)), hyper_parameter)

plt.xlabel('Number of Feature Maps')
plt.legend()
plt.ylabel('Testing Accuracy')
plt.title("Testing Accuracy Against Number of Feature Maps")
plt.show()
plt.savefig('./2_test_acc_3-2.png')