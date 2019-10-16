# Import required libraries
import math
import time
import pandas as pd
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

NUM_FEATURES = 7

learning_rate = 0.001
batch_size = 8
seed = 9
np.random.seed(seed)

DEBUG = False

# Create the model with 3 layers (1 hidden layer)
def model_3_layers(num_neurons, decay_parameter, trainX, trainY, debug=False):
    # Input layer
    x = tf.placeholder(tf.float32, shape=(None, trainX.shape[1]), name='inputs')
    y_ = tf.placeholder(tf.float32, shape=(None, 1), name='label')

    # Hidden layer
    hid1_size = 10
    W1 = tf.Variable(tf.truncated_normal([trainX.shape[1], hid1_size], stddev=1.0 / np.sqrt(NUM_FEATURES), seed=seed), name='W1', dtype=tf.float32)
    b1 = tf.Variable(tf.zeros([hid1_size]), name='b1', dtype=tf.float32)
    H1 = tf.nn.relu(tf.matmul(x, W1) + b1)

    # Output layer
    W2 = tf.Variable(tf.truncated_normal([hid1_size, 1], stddev=1.0 / np.sqrt(hid1_size), seed=seed), name='W2', dtype=tf.float32)
    b2 = tf.Variable(tf.zeros([1]), name='b2', dtype=tf.float32)
    Y = tf.matmul(H1, W2) + b2 

    # Compute regularization term
    regularization_term = decay_parameter * (tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2))
    loss = tf.reduce_mean(tf.square(y_ - Y)) + regularization_term
    
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(loss)
    
    if debug:
        return train_op, loss, {"x": x, "y_": y_, "W1": W1, "b1": b1, "H1": H1, "W2": W2, "b2": b2, "Y": Y, "regularization_term": regularization_term}
    else:
        return train_op, loss, {"x": x, "y_": y_}

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
def training(trainX, trainY, num_classes, decay_parameter, batch_size, no_of_layers, no_of_epochs, testX=None, testY=None,
             predict_values=False, predictX=None, predictY=None):
    with tf.Session() as sess:
        train_losses = []
        test_losses = []

        model, loss_function, feed_dict_values = model_3_layers(num_classes, decay_parameter, trainX, trainY, debug=predict_values)
        x, y_ = feed_dict_values["x"], feed_dict_values["y_"]

        # Reseed so shuffled data for mini-batch gradient descent are consistent with each training
        np.random.seed(seed)
            
        # Initialize weights and biases in the model chosen
        sess.run(tf.global_variables_initializer())
        
        # Get initial train and test loss
        train_losses.append(loss_function.eval(feed_dict={x: trainX, y_: trainY}))
        test_losses.append(loss_function.eval(feed_dict={x: testX, y_: testY}))
        
        for i in range(no_of_epochs):
            # Shuffle the dataset before grouping them into minibatches for gradient updates
            dataset_size = trainX.shape[0]
            idx = np.arange(dataset_size)
            np.random.shuffle(idx)
            trainX, trainY = trainX[idx], trainY[idx]

            # Create minibatches
            mini_batches = create_mini_batches(trainX, trainY, batch_size)
            
            for mini_batch in mini_batches:
                x_mini, y_mini = mini_batch
                model.run(feed_dict={x: x_mini, y_: y_mini})
            
            train_loss = loss_function.eval(feed_dict={x: trainX, y_: trainY})
            test_loss = loss_function.eval(feed_dict={x: testX, y_: testY})

            # Append training and test losses per epoch
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            
            if (i + 1) % 1000 == 0:
              print('Train loss after {} epochs is: {}'.format(i + 1, train_loss))
              print('Test loss after {} epochs is: {}'.format(i + 1, test_loss))
            
        if predict_values:    
          Y = feed_dict_values["Y"]
          Y_ = Y.eval(feed_dict={x: predictX, y_: predictY})
          return train_losses, test_losses, Y_
        else:
          return train_losses, test_losses

# Q1
# read and shuffle data
admit_data = np.genfromtxt('admission_predict.csv', delimiter= ',',dtype = np.float32)
X_data, Y_data = admit_data[1:,1:8], admit_data[1:,-1:]

idx = np.arange(X_data.shape[0])
np.random.shuffle(idx)
X_data, Y_data = X_data[idx], Y_data[idx]

# divide dataset into 70:30 ratio for training and testing
train_length = math.ceil(len(idx) * 0.7)

trainX, testX = X_data[:train_length], X_data[train_length:]
trainY, testY = Y_data[:train_length], Y_data[train_length:]

testX = (testX - np.mean(trainX, axis=0)) / np.std(trainX, axis=0)
trainX = (trainX - np.mean(trainX, axis=0)) / np.std(trainX, axis=0)

# Shuffle indexes and get first 50 random indexes for test data
test_idx = np.arange(testX.shape[0])
np.random.shuffle(test_idx)
X_data_50, Y_data_50 = testX[test_idx[:50]], testY[test_idx[:50]]

# Train and predict
epoch_no = 5000

training_losses, test_losses = training(trainX, trainY, num_classes=10, decay_parameter=10 ** -3, batch_size=8, no_of_layers=3, no_of_epochs=epoch_no, testX=testX, testY=testY)

# Plot learning curves
print('Final training loss after {} epochs is {}'.format(epoch_no, training_losses[-1]))
print('Final test loss is {} epochs is {}'.format(epoch_no, test_losses[-1]))

fig, ax = plt.subplots()
ax.plot(range(epoch_no + 1), training_losses, label='Training')
ax.plot(range(epoch_no + 1), test_losses, label='Test')
legend = ax.legend(loc='upper right')

plt.xlabel(str(epoch_no) +' iterations')
plt.ylabel('Error/loss')
plt.title('Learning Curve of 3-layer NN over {} epochs'.format(epoch_no))
plt.savefig('figures/project_1b_q1a_{}_epochs.png'.format(epoch_no))

fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.plot(range(100, epoch_no + 1, 100), [training_losses[epoch] for epoch in range(100, epoch_no + 1, 100)], 'xb-', label='Training')
ax2.plot(range(100, epoch_no + 1, 100), [test_losses[epoch] for epoch in range(100, epoch_no + 1, 100)], 'xr-', label='Test')
legend = ax2.legend(loc='upper right')

plt.xlabel(str(epoch_no) +' iterations')
plt.ylabel('Error/loss')
plt.title('Losses of 3-layer NN every 100 epochs')
plt.savefig('figures/project_1b_q1b.png')

training_losses, test_losses, model_outputs = training(trainX, trainY, num_classes=10, decay_parameter=10 ** -3, batch_size=8, no_of_layers=3, no_of_epochs=1000, testX=testX, testY=testY, predict_values=True, predictX=X_data_50,predictY=Y_data_50)

fig3, ax3 = plt.subplots(figsize=(20, 12))
ax3.plot([str(point_no) for point_no in range(1, 51)], model_outputs, 'xb-', label="Generated outputs")
ax3.plot([str(point_no) for point_no in range(1, 51)], Y_data_50, 'xr-', label="Desired outputs")
legend = ax3.legend(loc='lower left')

plt.xlabel('Point number')
plt.ylabel('Chance of admit')
plt.title('Generated vs Desired Outputs of Final Model for 50 points')
plt.savefig('figures/project_1b_q1c.png')

print(training_losses[0], test_losses[0])

