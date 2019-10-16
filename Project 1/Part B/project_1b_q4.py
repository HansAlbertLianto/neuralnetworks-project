import math
import time
import pandas as pd
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

learning_rate = 0.001
epochs = 1000
batch_size = 8
seed = 9
np.random.seed(seed)

# Create the model with 4 layers (2 hidden layers)
def model_4_layers(num_neurons, decay_parameter, trainX, trainY, num_features, dropout=False, debug=False):
    # Input layer
    x = tf.placeholder(tf.float32, shape=(None, trainX.shape[1]), name='inputs')
    y_ = tf.placeholder(tf.float32, shape=(None, 1), name='label')
    dropout_rate = tf.placeholder(tf.float32)

    # 1st Hidden layer
    hid1_size = 50
    W1 = tf.Variable(tf.truncated_normal([trainX.shape[1], hid1_size], stddev=1.0 / np.sqrt(num_features), seed=seed), name='W1', dtype=tf.float32)
    b1 = tf.Variable(tf.zeros([hid1_size]), name='b1', dtype=tf.float32)
    H1 = tf.nn.relu(tf.matmul(x, W1) + b1)
    if (dropout): H1 = tf.nn.dropout(H1, rate=dropout_rate)
    
    # 2nd Hidden layer
    hid2_size = 50
    W2 = tf.Variable(tf.truncated_normal([hid1_size, hid2_size], stddev=1.0 / np.sqrt(hid1_size), seed=seed), name='W2', dtype=tf.float32)
    b2 = tf.Variable(tf.zeros([hid2_size]), name='b2', dtype=tf.float32)
    H2 = tf.nn.relu(tf.matmul(H1, W2) + b2)
    if (dropout): H2 = tf.nn.dropout(H2, rate=dropout_rate)

    # Output layer
    W3 = tf.Variable(tf.truncated_normal([hid2_size, 1], stddev=1.0 / np.sqrt(hid2_size), seed=seed), name='W3', dtype=tf.float32)
    b3 = tf.Variable(tf.zeros([1]), name='b3', dtype=tf.float32)
    Y = tf.matmul(H2, W3) + b3

    # Compute regularization term
    regularization_term = decay_parameter * (tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(W3))
    loss = tf.reduce_mean(tf.square(y_ - Y)) + regularization_term
    
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(loss)
    
    if debug:
        return train_op, loss, {"x": x, "y_": y_, "dropout_rate": dropout_rate, "W1": W1, "b1": b1, "H1": H1, "W2": W2, "b2": b2, "H2": H2, "W3": W3, "b3": b3, "Y": Y, "regularization_term": regularization_term}
    else:
        return train_op, loss, {"x": x, "y_": y_, "dropout_rate": dropout_rate}
      
# Create the model with 4 layers (2 hidden layers)
def model_5_layers(num_neurons, decay_parameter, trainX, trainY, num_features, dropout=False, debug=False):
    # Input layer
    x = tf.placeholder(tf.float32, shape=(None, trainX.shape[1]), name='inputs')
    y_ = tf.placeholder(tf.float32, shape=(None, 1), name='label')
    dropout_rate = tf.placeholder(tf.float32)

    # 1st Hidden layer
    hid1_size = 50
    W1 = tf.Variable(tf.truncated_normal([trainX.shape[1], hid1_size], stddev=1.0 / np.sqrt(num_features), seed=seed), name='W1', dtype=tf.float32)
    b1 = tf.Variable(tf.zeros([hid1_size]), name='b1', dtype=tf.float32)
    H1 = tf.nn.relu(tf.matmul(x, W1) + b1)
    if (dropout): H1 = tf.nn.dropout(H1, rate=dropout_rate)
    
    # 2nd Hidden layer
    hid2_size = 50
    W2 = tf.Variable(tf.truncated_normal([hid1_size, hid2_size], stddev=1.0 / np.sqrt(hid1_size), seed=seed), name='W2', dtype=tf.float32)
    b2 = tf.Variable(tf.zeros([hid2_size]), name='b2', dtype=tf.float32)
    H2 = tf.nn.relu(tf.matmul(H1, W2) + b2)
    if (dropout): H2 = tf.nn.dropout(H2, rate=dropout_rate)
    
    # 3rd Hidden layer
    hid3_size = 50
    W3 = tf.Variable(tf.truncated_normal([hid2_size, hid3_size], stddev=1.0 / np.sqrt(hid2_size), seed=seed), name='W3', dtype=tf.float32)
    b3 = tf.Variable(tf.zeros([hid3_size]), name='b3', dtype=tf.float32)
    H3 = tf.nn.relu(tf.matmul(H2, W3) + b3)
    if (dropout): H3 = tf.nn.dropout(H3, rate=dropout_rate)

    # Output layer
    W4 = tf.Variable(tf.truncated_normal([hid3_size, 1], stddev=1.0 / np.sqrt(hid3_size), seed=seed), name='W4', dtype=tf.float32)
    b4 = tf.Variable(tf.zeros([1]), name='b4', dtype=tf.float32)
    Y = tf.matmul(H3, W4) + b4

    # Compute regularization term
    regularization_term = decay_parameter * (tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(W3) + tf.nn.l2_loss(W4))
    loss = tf.reduce_mean(tf.square(y_ - Y)) + regularization_term
    
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(loss)
    
    if debug:
        return train_op, loss, {"x": x, "y_": y_, "dropout_rate": dropout_rate, "W1": W1, "b1": b1, "H1": H1, "W2": W2, "b2": b2, "H2": H2, "W3": W3, "b3": b3, "Y": Y, "regularization_term": regularization_term}
    else:
        return train_op, loss, {"x": x, "y_": y_, "dropout_rate": dropout_rate}
      
def create_mini_batches(X, y, batch_size): 
    mini_batches = [] 
    data = np.hstack((X, y)) 
    # np.random.shuffle(data) 
    n_minibatches = data.shape[0] // batch_size 
    i = 0
  
    for i in range(n_minibatches + 1): 
        mini_batch = data[i * batch_size:(i + 1)*batch_size, :] 
        X_mini = mini_batch[:, :-1] 
        Y_mini = mini_batch[:, -1].reshape((-1, 1)) 
        mini_batches.append((X_mini, Y_mini)) 
    if data.shape[0] % batch_size != 0: 
        mini_batch = data[i * batch_size:data.shape[0]] 
        X_mini = mini_batch[:, :-1] 
        Y_mini = mini_batch[:, -1].reshape((-1, 1)) 
        mini_batches.append((X_mini, Y_mini)) 
    return mini_batches 
  

# Training function, model choices and hyperparameter choices, including batch size for minibatch gradient descent
# are chosen here.
def training(trainX, trainY, num_classes, decay_parameter, batch_size, no_of_layers, epochs, testX=None, testY=None,
             predict_values=False, predictX=None, predictY=None, dropout=False):
    with tf.Session() as sess:
        train_losses = []
        test_losses = []

        if(no_of_layers == 4):
          model, loss_function, feed_dict_values = model_4_layers(num_classes, decay_parameter, trainX, trainY, num_features=trainX.shape[1], dropout=dropout, debug=True)
        elif(no_of_layers == 5):
          model, loss_function, feed_dict_values = model_5_layers(num_classes, decay_parameter, trainX, trainY, num_features=trainX.shape[1], dropout=dropout, debug=True)
        x, y_, dropout_rate = feed_dict_values["x"], feed_dict_values["y_"], feed_dict_values["dropout_rate"]

        # Reseed so shuffled data for mini-batch gradient descent are consistent with each training
        np.random.seed(seed)
            
        # Initialize weights and biases in the model chosen
        sess.run(tf.global_variables_initializer())
        for i in range(epochs):
            # Shuffle the dataset before grouping them into minibatches for gradient updates
            dataset_size = trainX.shape[0]
            idx = np.arange(dataset_size)
            np.random.shuffle(idx)
            trainX, trainY = trainX[idx], trainY[idx]

            # Create minibatches
            mini_batches = create_mini_batches(trainX, trainY, batch_size)
            
            for mini_batch in mini_batches:
                x_mini, y_mini = mini_batch
                model.run(feed_dict={x: x_mini, y_: y_mini, dropout_rate: 0.2})
            
            train_loss = loss_function.eval(feed_dict={x: trainX, y_: trainY, dropout_rate: 0})
            test_loss = loss_function.eval(feed_dict={x: testX, y_: testY, dropout_rate: 0})
            
            if (i + 1) == epochs:
              print("Regularization term = {}".format(feed_dict_values["regularization_term"].eval(feed_dict={x: testX, y_: testY, dropout_rate: 0})))

            # Append training and test losses per epoch
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            
        if predict_values:    
          Y = feed_dict_values["Y"]
          Y_ = Y.eval(feed_dict={x: predictX, y_: predictY})
          return train_losses, test_losses, Y_
        else:
          return train_losses, test_losses
        
# helper to divide train and test data
def split_data_to_train_and_test(data, no_of_columns, train_percentage=0.7):
  X_data, Y_data = data[:,1:no_of_columns], data[:,-1:]
  
  idx = np.arange(X_data.shape[1])

  # divide dataset into 70:30 ratio for training and testing
  train_length = math.ceil(len(idx) * train_percentage)

  trainX = X_data[:train_length]
  trainY = Y_data[:train_length]
  testX = X_data[train_length:]
  testY = Y_data[train_length:]
  
  testX = (testX - np.mean(trainX, axis=0)) / np.std(trainX, axis=0)
  trainX = (trainX - np.mean(trainX, axis=0)) / np.std(trainX, axis=0)
  
  return trainX, trainY, testX, testY

def train_and_save_figures(trainX, trainY, testX, testY, dropout=False, no_of_layers=4, original=False):
  # Train and predict
  training_losses, test_losses = training(trainX, trainY, num_classes=10, epochs=epochs, decay_parameter=10 ** -3, batch_size=8, no_of_layers=no_of_layers, testX=testX, testY=testY, dropout=dropout)

  # Plot learning curves
  print('=========== {} layers, dropout: {} =========='.format(no_of_layers, dropout))
  print('Final training loss is {}'.format(training_losses[-1]))
  print('Final test loss is {}'.format(test_losses[-1]))
  print('=============================================')

  fig, ax = plt.subplots()
  ax.plot(range(epochs), training_losses, label='Training')
  ax.plot(range(epochs), test_losses, label='Test')
  legend = ax.legend(loc='upper right')

  plt.xlabel(str(epochs) +' iterations')
  plt.ylabel('Loss')

  q4_prefix = 'project_1b_q4'
  
  if (no_of_layers == 4):
    title = 'Learning Curve of 4-layers NN'
    if (dropout):
      title += ' with dropout'
    plt.title(title)
    plt.savefig('figures/{}_{}.png'.format(q4_prefix, title))
  elif (no_of_layers == 5):
    title = 'Learning Curve of 5-layers NN'
    if (dropout):
      title += ' with dropout'
    plt.title(title)
    plt.savefig('figures/{}_{}.png'.format(q4_prefix, title))
  
  return training_losses, test_losses

original_df = pd.read_csv('admission_predict.csv')
original_df = original_df.drop('CGPA', 1)
idx = np.arange(original_df.to_numpy().shape[0])
initial_data = original_df.to_numpy()
initial_shuffled_data = initial_data[idx]
original_trainX, original_trainY, original_testX, original_testY = split_data_to_train_and_test(initial_shuffled_data, 8)

# 4 layers without dropout
four_layers_training_losses, four_layers_test_losses = train_and_save_figures(original_trainX, original_trainY, original_testX, original_testY, original=True)

# 4 layers with dropout
four_layers_training_losses, four_layers_test_losses = train_and_save_figures(original_trainX, original_trainY, original_testX, original_testY, original=True, dropout=True)

# 5 layers without dropout
five_layers_training_losses, five_layers_test_losses = train_and_save_figures(original_trainX, original_trainY, original_testX, original_testY, no_of_layers=5, original=True)

# 5 layers with dropout
five_layers_training_losses, five_layers_test_losses = train_and_save_figures(original_trainX, original_trainY, original_testX, original_testY, no_of_layers=5, original=True, dropout=True)