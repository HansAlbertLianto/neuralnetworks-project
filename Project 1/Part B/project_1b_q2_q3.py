# Import required libraries
import math
import time
import pandas as pd
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

learning_rate = 0.001
epochs = 1000
batch_size = 8
seed = 9
np.random.seed(seed)

# Shuffle the dataset before test/train split
original_df = pd.read_csv('admission_predict.csv')

idx = np.arange(original_df.to_numpy().shape[0])
np.random.shuffle(idx)

shuffled_df = original_df.iloc[idx]
train_length = math.ceil(len(idx) * 0.7)
train_df = shuffled_df.iloc[:train_length]

corr_df = train_df.drop('Serial No.', 1)
corr = corr_df.corr()
fig, ax = plt.subplots(figsize=(10, 10))

for (i, j), z in np.ndenumerate(corr):
    ax.text(j, i, '{:0.3f}'.format(z), ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))


ax.matshow(corr)
plt.xticks(range(len(corr.columns)), corr.columns, rotation='vertical');
plt.yticks(range(len(corr.columns)), corr.columns);
plt.savefig('figures/project_1b_q2_feature_correlation_matrix.png')

# find highly correlated features, above 0.8
correlated_features = set()
corr = train_df.drop('Chance of Admit', axis=1).corr()

for i in range(len(corr.columns)):
    for j in range(i):
        if abs(corr.iloc[i, j]) > 0.8:
            colname = corr.columns[i]
            correlated_features.add(colname)

print(correlated_features)

# Create the model with 3 layers (1 hidden layer)
def model_3_layers(num_neurons, decay_parameter, trainX, trainY, num_features, debug=False):
    # Input layer
    x = tf.placeholder(tf.float32, shape=(None, trainX.shape[1]), name='inputs')
    y_ = tf.placeholder(tf.float32, shape=(None, 1), name='label')

    # Hidden layer
    hid1_size = 10
    W1 = tf.Variable(tf.truncated_normal([trainX.shape[1], hid1_size], stddev=1.0 / np.sqrt(num_features), seed=seed), name='W1', dtype=tf.float32)
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
def training(trainX, trainY, num_classes, decay_parameter, batch_size, no_of_layers, epochs, testX=None, testY=None,
             predict_values=False, predictX=None, predictY=None):
    with tf.Session() as sess:
        train_losses = []
        test_losses = []

        model, loss_function, feed_dict_values = model_3_layers(num_classes, decay_parameter, trainX, trainY, num_features=trainX.shape[1])
        x, y_ = feed_dict_values["x"], feed_dict_values["y_"]

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
                model.run(feed_dict={x: x_mini, y_: y_mini})
            
            train_loss = loss_function.eval(feed_dict={x: trainX, y_: trainY})
            test_loss = loss_function.eval(feed_dict={x: testX, y_: testY})

            # Append training and test losses per epoch
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            
        if predict_values:    
          Y = feed_dict_values["Y"]
          Y_ = Y.eval(feed_dict={x: predictX, y_: predictY})
          return train_losses, test_losses, Y_
        else:
          return train_losses, test_losses

# remove features one by one, keep track of the loss
features = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR', 'CGPA',
       'Research']

# helper to divide train and test data
def split_data_to_train_and_test(data, no_of_columns, train_percentage=0.7):
  X_data, Y_data = data[:,1:no_of_columns], data[:,-1:]

  # divide dataset into 70:30 ratio for training and testing
  train_length = math.ceil(len(idx) * train_percentage)

  trainX = X_data[:train_length]
  trainY = Y_data[:train_length]
  testX = X_data[train_length:]
  testY = Y_data[train_length:]
  
  testX = (testX - np.mean(trainX, axis=0)) / np.std(trainX, axis=0)
  trainX = (trainX - np.mean(trainX, axis=0)) / np.std(trainX, axis=0)
  
  return trainX, trainY, testX, testY

def train_and_save_figures(trainX, trainY, testX, testY, features_removed, original=False):
  # Train and predict
  training_losses, test_losses = training(trainX, trainY, num_classes=10, epochs=epochs, decay_parameter=10 ** -3, batch_size=8,
                                                         no_of_layers=3, testX=testX, testY=testY)

  # Plot learning curves
  if len(features_removed) == 0:
    print('Features removed: None')
  else:
    print('Features removed: {}'.format(', '.join(features_removed)))

  print('Final training loss is {}'.format(training_losses[-1]))
  print('Final test loss is {}'.format(test_losses[-1]))
  print('=============================================')

  fig, ax = plt.subplots()
  ax.plot(range(epochs), training_losses, label='Training')
  ax.plot(range(epochs), test_losses, label='Test')
  legend = ax.legend(loc='upper right')

  plt.xlabel(str(epochs) +' iterations')
  plt.ylabel('Loss')
  
  features_removed_str = ', '.join(features_removed) if len(features_removed) > 0 else 'none'
  features_removed_fig_filename = '-'.join(features_removed) if len(features_removed) > 0 else 'none'
  
  if original:
    plt.title('Learning Curve of 3-layer NN, removing features {} - {}'.format(features_removed_str, 'original'))
    plt.savefig('figures/project_1b_q3_{}_removed_original.png'.format(features_removed_fig_filename))
  else:
    plt.title('Learning Curve of 3-layer NN, removing features {}'.format(features_removed_str)) 
    plt.savefig('figures/project_1b_q3_{}_removed.png'.format(features_removed_fig_filename))
  
  return training_losses, test_losses

# trainX, trainY, testX, testY = split_data_to_train_and_test(original_df.to_numpy(), 9)
# # train model and keep the loss
# training_losses, test_losses = train_and_save_figures(trainX, trainY, testX, testY, 'NOTHING')

initial_data = original_df.to_numpy()
initial_shuffled_data = initial_data[idx]
original_trainX, original_trainY, original_testX, original_testY = split_data_to_train_and_test(initial_shuffled_data, len(features) + 1)
original_training_losses, original_test_losses = train_and_save_figures(original_trainX, original_trainY, original_testX, original_testY,
                                                                        features_removed=[], original=True)
original_best_final_test_loss = original_test_losses[-1]

def rfe(df, feature_list, prev_best_final_test_loss, removed_features=[]):
  final_test_losses = []
  
  for feature in feature_list:
    rfe_df = df.drop(feature, 1)
    admit_data = rfe_df.to_numpy()
    admit_shuffled_data = admit_data[idx]
    trainX, trainY, testX, testY = split_data_to_train_and_test(admit_shuffled_data, len(feature_list) + 1)
    
    # train model and keep the loss
    training_losses, test_losses = train_and_save_figures(trainX, trainY, testX, testY, removed_features + [feature])
    
    final_test_losses.append(test_losses[-1])
  
  best_test_loss = min(final_test_losses)
  best_feature = feature_list[final_test_losses.index(best_test_loss)]
  
  # If better than previous model, recurse; otherwise return the previous model
  if best_test_loss < prev_best_final_test_loss:
    removed_features.append(best_feature)
    feature_list.remove(best_feature)
    return rfe(df.drop(best_feature, 1), feature_list, best_test_loss, removed_features=removed_features)
  else:
    return prev_best_final_test_loss, removed_features
  
print(rfe(original_df, feature_list=features, prev_best_final_test_loss=original_best_final_test_loss))

