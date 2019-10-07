# Import required libraries
import math
import time
import pandas as pd
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Declare hyperparameters and model parameters
NUM_FEATURES = 21
NUM_CLASSES = 3

learning_rate = 0.01
epochs = 5000
batch_size = 32
num_neurons = 10
seed = 10
l2_beta = 10 ** -6

# Set initial seed so values are consistent
np.random.seed(seed)

DEBUG = False

# Function to scale features
def scale(X, X_min, X_max):
    return (X - X_min) / (X_max - X_min)

# Read train data
train_input = np.genfromtxt('ctg_data_cleaned.csv', delimiter= ',')

# Generate feature-scaled training data as input to model
X, Y_ = train_input[1:, :21], train_input[1:,-1].astype(int)
X = scale(X, np.min(X, axis=0), np.max(X, axis=0))

# Generate one-hot output as desired output of model
# Y_ is a list of classes, Y is the one-hot encoded version of Y_
Y = np.zeros((Y_.shape[0], NUM_CLASSES))
Y[np.arange(Y_.shape[0]), Y_ - 1] = 1 # One-hot matrix

# Create the model with 3 layers (1 hidden layer)
def model_3_layers(num_neurons, decay_parameter, debug=False):
    x = tf.placeholder(tf.float32, [None, NUM_FEATURES])
    y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])

    # Build the graph for the deep net
    W1 = tf.Variable(tf.truncated_normal([NUM_FEATURES, num_neurons], stddev=1.0/math.sqrt(float(NUM_FEATURES)), seed=seed), name='W1')
    b1 = tf.Variable(tf.zeros([num_neurons]), name='b1')
    H1 = tf.nn.relu(tf.matmul(x, W1) + b1)

    W2 = tf.Variable(tf.truncated_normal([num_neurons, NUM_CLASSES], stddev=1.0/math.sqrt(float(num_neurons)), seed=seed), name='W2')
    b2 = tf.Variable(tf.zeros([NUM_CLASSES]), name='b2')
    logits = tf.matmul(H1, W2) + b2
    Y = tf.nn.softmax(logits)

    # Compute regularization term
    regularization_term = decay_parameter * (tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2))
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=logits)
    loss = tf.reduce_mean(cross_entropy) + regularization_term
    
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(loss)

    correct_prediction = tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1)), tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)
    
    if debug:
        return train_op, loss, accuracy, {"x": x, "y_": y_, "W1": W1, "b1": b1, "H1": H1, "W2": W2, "b2": b2, 
                                          "logits": logits, "Y": Y, "regularization_term": regularization_term}
    else:
        return train_op, loss, accuracy, {"x": x, "y_": y_}

# Question 5
# Neural Network with 4 layers

# Create the model with 4 layers (2 hidden layers)
def model_4_layers(num_neurons, decay_parameter, debug=False):
    # Create the model
    x = tf.placeholder(tf.float32, [None, NUM_FEATURES])
    y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])

    # Build the graph for the deep net
    W1 = tf.Variable(tf.truncated_normal([NUM_FEATURES, num_neurons], stddev=1.0/math.sqrt(float(NUM_FEATURES)), seed=seed), name='W1')
    b1 = tf.Variable(tf.zeros([num_neurons]), name='b1')
    H1 = tf.nn.relu(tf.matmul(x, W1) + b1)

    W2 = tf.Variable(tf.truncated_normal([num_neurons, num_neurons], stddev=1.0/math.sqrt(float(num_neurons)), seed=seed), name='W2')
    b2 = tf.Variable(tf.zeros([num_neurons]), name='b2')
    H2 = tf.nn.relu(tf.matmul(H1, W2) + b2)
    
    W3 = tf.Variable(tf.truncated_normal([num_neurons, NUM_CLASSES], stddev=1.0/math.sqrt(float(num_neurons)), seed=seed), name='W3')
    b3 = tf.Variable(tf.zeros([NUM_CLASSES]), name='b3')
    logits = tf.matmul(H2, W3) + b3
    Y = tf.nn.softmax(logits)

    # Compute regularization term
    regularization_term = decay_parameter * (tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(W3))
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=logits)
    loss = tf.reduce_mean(cross_entropy) + regularization_term
    
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(loss)

    correct_prediction = tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1)), tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)
    
    if debug:
        return train_op, loss, accuracy, {"x": x, "y_": y_, "W1": W1, "b1": b1, "H1": H1, "W2": W2, "b2": b2, "H2": H2,
                                          "W3": W3, "b3": b3, "logits": logits, "Y": Y,
                                          "regularization_term": regularization_term}
    else:
        return train_op, loss, accuracy, {"x": x, "y_": y_}

# Training function, model choices and hyperparameter choices, including batch size for minibatch gradient descent
# are chosen here.
def training(trainX, trainY, num_classes, decay_parameter, batch_size, no_of_layers, testX=None, testY=None,
             use_small_dataset=True):
    with tf.Session() as sess:
        train_acc = []
        test_acc = []
        losses = []
        
        if use_small_dataset:
            # Get model with debug values
            if no_of_layers == 3:
                model, loss, accuracy, debug_values = model_3_layers(num_classes, decay_parameter, debug=True)
                x, y_, W1, b1, H1, W2, b2, logits, Y, regularization_term = debug_values["x"], debug_values["y_"], debug_values["W1"], debug_values["b1"], debug_values["H1"], debug_values["W2"], debug_values["b2"], debug_values["logits"], debug_values["Y"], debug_values["regularization_term"]
            elif no_of_layers == 4:
                model, loss, accuracy, debug_values = model_4_layers(num_classes, decay_parameter, debug=True)
                x, y_, W1, b1, H1, W2, b2, H2, W3, b3, logits, Y, regularization_term = debug_values["x"], debug_values["y_"], debug_values["W1"], debug_values["b1"], debug_values["H1"], debug_values["W2"], debug_values["b2"], debug_values["H2"], debug_values["W3"], debug_values["b3"], debug_values["logits"], debug_values["Y"], debug_values["regularization_term"]
        else:
            # Get model with feed_dict values
            if no_of_layers == 3:
                model, loss, accuracy, feed_dict_values = model_3_layers(num_classes, decay_parameter)
                x, y_ = feed_dict_values["x"], feed_dict_values["y_"]
            elif no_of_layers == 4:
                model, loss, accuracy, feed_dict_values = model_4_layers(num_classes, decay_parameter)
                x, y_ = feed_dict_values["x"], feed_dict_values["y_"]
                
            # Reseed so shuffled data for mini-batch gradient descent are consistent with each training
            np.random.seed(seed)
            
        # Initialize weights and biases in the model chosen
        sess.run(tf.global_variables_initializer())
        
        for i in range(epochs):
            if use_small_dataset:
                # Debugging
                model.run(feed_dict={x_: trainX, y_: trainY})
                train_acc.append(accuracy.eval(feed_dict={x: trainX, y_: trainY}))

                if i == 0 or i == epochs - 1:
                    x_, y__, logits_, W1_, b1_, H1_, W2_, b2_, regularization_term_, Y_ = sess.run([x, y_, logits, W1, b1, H1, W2, b2, regularization_term, Y], feed_dict={x: trainX, y_: trainY})
                    print('iteration %d:' % (i))
                    print('X: ', x_)
                    print('Y: ', y__)
                    print('W1: ', W1_)
                    print('b1:', b1_)
                    print('H1: ', H1_)
                    print('W2: ', W2_)
                    print('b2: ', b2_)
                    print('Logits: ', logits_)
                    print('Y: ', Y_)
                    print('Regularization term: ', regularization_term_)

                if i % 100 == 0:
                    print('iter %d: accuracy %g'%(i, train_acc[i]))
            else:
                # Shuffle the dataset before grouping them into minibatches for gradient updates
                dataset_size = trainX.shape[0]
                idx = np.arange(dataset_size)
                np.random.shuffle(idx)
                trainX, trainY = trainX[idx], trainY[idx]
                
                # Initialize initial average epoch training accuracy
                avg_epoch_train_acc = 0
                no_of_minibatches = math.ceil(dataset_size / batch_size)

                # Mini-batch gradient descent
                for start, end in zip(range(0, dataset_size, batch_size), range(batch_size, dataset_size, batch_size)):
                    model.run(feed_dict={x: trainX[start:end], y_: trainY[start:end]})
                    avg_epoch_train_acc += accuracy.eval(feed_dict={x: trainX[start:end], y_: trainY[start:end]}) / no_of_minibatches
                    
                # Append training and test accuracies per epoch
                train_acc.append(avg_epoch_train_acc)
                test_acc.append(accuracy.eval(feed_dict={x: testX, y_: testY}))
                losses.append(loss.eval(feed_dict={x: trainX, y_: trainY}))
                    
                if i % 100 == 0:
                    print('iter %d: accuracy %g, test accuracy %g, loss %g'%(i, train_acc[i], test_acc[i], losses[i]))

        return train_acc, test_acc, losses



# Decide to use small dataset or whole dataset with cross-validation techniques
if DEBUG:
    # Get first 1000 tuples.
    trainX = X[:1000]
    trainY = Y[:1000]
    n = trainX.shape[0]
    train_acc, test_acc, losses = training(trainX, trainY, num_classes=10, decay_parameter=10 ** -6,
                                           batch_size=32, no_of_layers=3)
else:
    # QUESTION 1
    # Shuffle the dataset.
    dataset_size = X.shape[0]
    benchmark_idx = np.arange(dataset_size)
    np.random.shuffle(benchmark_idx)
    X_shuffled, Y_shuffled = X[benchmark_idx], Y[benchmark_idx]
    
    # Split the dataset to 70:30; 70 for training and 30 for testing
    idx_split = math.ceil(X_shuffled.shape[0] / 5 * 0.7) * 5
    
    # cv_and_trainX and cv_and_trainY will continually be used in later questions for hyperparameter tuning
    cv_and_trainX, testX = X_shuffled[:idx_split], X_shuffled[idx_split:]
    cv_and_trainY, testY = Y_shuffled[:idx_split], Y_shuffled[idx_split:]

    train_acc, test_acc, losses = training(cv_and_trainX, cv_and_trainY, num_classes=10, decay_parameter=10 ** -6, batch_size=32,
                                  no_of_layers=3, testX=testX, testY=testY, use_small_dataset=False)

# Plot learning curves
print('Final training accuracy is {}'.format(train_acc[-1]))
print('Final test accuracy is {}'.format(test_acc[-1]))

fig, ax = plt.subplots()
ax.plot(range(epochs), train_acc, label='Training')
ax.plot(range(epochs), test_acc, label='Test')
legend = ax.legend(loc='lower right')

plt.xlabel(str(epochs) +' iterations')
plt.ylabel('Accuracy')
plt.title('Learning Curve of 3-layer NN with untuned hyperparameters')
plt.savefig('figures/project_1a_q1a.png')

fig, ax = plt.subplots()
ax.plot(range(epochs)[1000:3001], train_acc[1000:3001], label='Training')
ax.plot(range(epochs)[1000:3001], test_acc[1000:3001], label='Test')
legend = ax.legend(loc='lower right')

plt.xlabel(str(epochs) +' iterations')
plt.ylabel('Accuracy')
plt.title('Learning Curve of 3-layer NN with untuned hyperparameters [A]')
plt.savefig('figures/project_1a_q1b_1.png')

fig, ax = plt.subplots()
ax.plot(range(epochs)[0:300], train_acc[0:300], label='Training')
ax.plot(range(epochs)[0:300], test_acc[0:300], label='Test')
legend = ax.legend(loc='lower right')

plt.xlabel(str(epochs) +' iterations')
plt.ylabel('Accuracy')
plt.title('Learning Curve of 3-layer NN with untuned hyperparameters [B]')
plt.savefig('figures/project_1a_q1b_2.png')

# Plot learning curves
print('Final loss is {}'.format(losses[-1]))
plt.clf()
plt.plot(range(epochs), losses)
plt.xlabel('Epoch number')
plt.ylabel('Average epoch loss')
plt.title('Loss of 3-layer NN with untuned hyperparameters')
plt.savefig('figures/project_1a_q1b_3.png')

# QUESTION 2
# Optimal batch size
BATCH_SIZES = [4, 8, 16, 32, 64]

batch_size_train_accs = []
batch_size_test_accs = []
batch_losses = []
batch_times = []

# Try training with different batch sizes
for batch_size in BATCH_SIZES:
    print("==============")
    print("USING BATCH SIZE = {}".format(batch_size))
    print("==============")
    
    train_accs = []
    cross_validation_accs = []
    losses = []
    total_time_taken = 0
    
    # Further split trainX and trainY to do 5-fold cross-validation
    for fold in range(5):
        print("==============")
        print("FOLD {}".format(fold + 1))
        print("==============")

        # Find indexes to split dataset further.
        start_idx = fold * int(idx_split / 5)
        end_idx = (fold + 1) * int(idx_split / 5)

        # Split training dataset further to training and test set.
        trainX = np.append(cv_and_trainX[:start_idx], cv_and_trainX[end_idx:], axis=0)
        trainY = np.append(cv_and_trainY[:start_idx], cv_and_trainY[end_idx:], axis=0)
        cv_X = cv_and_trainX[start_idx:end_idx]
        cv_Y = cv_and_trainY[start_idx:end_idx]

        # Train the model.
        start_time = time.time()
        train_acc, cv_acc, loss = training(trainX, trainY, num_classes=10, decay_parameter=10 ** -6, batch_size=batch_size,
                                           no_of_layers=3, testX=cv_X, testY=cv_Y, use_small_dataset=False)
        time_taken = time.time() - start_time
        
        # Record cross-validation accuracies of model and total time taken per epoch
        train_accs.append(train_acc)
        cross_validation_accs.append(cv_acc)
        losses.append(loss)
        total_time_taken += time_taken / epochs

    # Find mean model training and cross-validation accuracies per epoch and average time taken for different batch sizes.
    model_train_accs_avg = np.average(np.array(train_accs), axis=0).tolist()
    model_test_accs_avg = np.average(np.array(cross_validation_accs), axis=0).tolist()
    losses_avg = np.average(np.array(losses), axis=0).tolist()
    
    batch_size_train_accs.append(model_train_accs_avg)
    batch_size_test_accs.append(model_test_accs_avg)
    batch_losses.append(losses_avg)
    batch_times.append(total_time_taken / 5)

# Plot accuracy
final_train_acc = []
final_test_acc = []

plt.clf()
fig, ax = plt.subplots()
for idx, batch_size in enumerate(BATCH_SIZES):
    ax.plot(range(epochs), batch_size_test_accs[idx], label="Batch size = " + str(batch_size))
    print("Batch size {} reached train accuracy of {} and test accuracy of {} after 5000 epochs."
          .format(batch_size, batch_size_train_accs[idx][-1], batch_size_test_accs[idx][-1]))
    
    final_train_acc.append(batch_size_train_accs[idx][-1])
    final_test_acc.append(batch_size_test_accs[idx][-1])
legend = ax.legend(loc='lower right')

plt.xlabel(str(epochs) +' iterations')
plt.ylabel('Accuracy')
plt.title('Learning curves of 3-layer NNs with different batch sizes')
plt.savefig('figures/project_1a_q2a_1.png')

plt.clf()
fig2, ax2 = plt.subplots()
ax2.plot([str(batch_size) for batch_size in BATCH_SIZES], final_train_acc, 'xb-', label="Final Train Accuracy")
ax2.plot([str(batch_size) for batch_size in BATCH_SIZES], final_test_acc, 'xr-', label="Final Test Accuracy")
legend = ax2.legend(loc='upper right')

plt.xlabel('Batch size')
plt.ylabel('Accuracy')
plt.title('Accuracies of 3-layer NNs with different batch sizes')
plt.savefig('figures/project_1a_q2a_2.png')

# Plot loss
plt.clf()
fig, ax = plt.subplots()
for idx, batch_size in enumerate(BATCH_SIZES):
    ax.plot(range(epochs), batch_losses[idx], label="Batch size = " + str(batch_size))
    print("Batch size {} reached a loss of {} after 5000 epochs.".format(batch_size, batch_losses[idx][-1]))
legend = ax.legend(loc='upper right')

plt.xlabel(str(epochs) +' iterations')
plt.ylabel('Loss')
plt.title('Losses of 3-layer NNs with diff. batch sizes')
plt.savefig('figures/project_1a_q2a_3.png')

# Plot the time taken
print(batch_times)
plt.clf()
plt.plot([str(batch_size) for batch_size in BATCH_SIZES], batch_times)
plt.xlabel('Batch size')
plt.ylabel('Average time taken per epoch')
plt.title('Duration per epoch with diff. batch sizes')
plt.savefig('figures/project_1a_q2a_4.png')

plt.clf()
plt.plot(BATCH_SIZES, batch_times, 'xb-')
plt.xlabel('Batch size')
plt.ylabel('Average time taken per epoch')
plt.title('Duration per epoch with diff. batch sizes [quantitative]')
plt.savefig('figures/project_1a_q2a_5.png')

# Train with optimal batch size
OPTIMAL_BATCH_SIZE = 16

train_acc, test_acc, loss = training(cv_and_trainX, cv_and_trainY, num_classes=10, decay_parameter=10 ** -6,
                                     batch_size=OPTIMAL_BATCH_SIZE, no_of_layers=3, testX=testX, testY=testY,
                                     use_small_dataset=False)

# Plot learning curves
print('Final training accuracy is {}'.format(train_acc[-1]))
print('Final test accuracy is {}'.format(test_acc[-1]))
print('Final loss is {}'.format(loss[-1]))

plt.clf()
fig, ax = plt.subplots()
ax.plot(range(epochs), train_acc, label='Training')
ax.plot(range(epochs), test_acc, label='Test')
legend = ax.legend(loc='lower right')

plt.xlabel(str(epochs) +' iterations')
plt.ylabel('Accuracy')
plt.title('Learning curve of 3-layer NN with batch size {}'.format(OPTIMAL_BATCH_SIZE))
plt.savefig('figures/project_1a_q2c.png')

# QUESTION 3
# Optimal number of hidden-layer neurons

NEURON_NUMS = [5, 10, 15, 20, 25]

num_neuron_train_accs = []
num_neuron_test_accs = []
num_neuron_losses = []

# Try training with different batch sizes
for neuron_num in NEURON_NUMS:
    print("==============")
    print("USING {} NUMBER OF NEURONS IN HIDDEN LAYER".format(neuron_num))
    print("==============")
    train_accs = []
    cross_validation_accs = []
    losses = []
    
    # Further split trainX and trainY to do 5-fold cross-validation
    for fold in range(5):
        print("==============")
        print("FOLD {}".format(fold + 1))
        print("==============")

        # Find indexes to split dataset further.
        start_idx = fold * int(idx_split / 5)
        end_idx = (fold + 1) * int(idx_split / 5)

        # Split training dataset further to training and test set.
        trainX = np.append(cv_and_trainX[:start_idx], cv_and_trainX[end_idx:], axis=0)
        trainY = np.append(cv_and_trainY[:start_idx], cv_and_trainY[end_idx:], axis=0)
        cv_X = cv_and_trainX[start_idx:end_idx]
        cv_Y = cv_and_trainY[start_idx:end_idx]

        # Train the model.
        train_acc, cv_acc, loss = training(trainX, trainY, num_classes=neuron_num, decay_parameter=10 ** -6,
                                           batch_size=OPTIMAL_BATCH_SIZE, no_of_layers=3, testX=cv_X, testY=cv_Y,
                                           use_small_dataset=False)
        
        # Record cross-validation accuracies of model
        train_accs.append(train_acc)
        cross_validation_accs.append(cv_acc)
        losses.append(loss)

    # Find mean model training accuracies per epoch and average time taken for differnt batch sizes.
    model_train_accs_avg = np.average(np.array(train_accs), axis=0).tolist()
    model_test_accs_avg = np.average(np.array(cross_validation_accs), axis=0).tolist()
    num_neuron_loss_avg = np.average(np.array(losses), axis=0).tolist()
    
    num_neuron_train_accs.append(model_train_accs_avg)
    num_neuron_test_accs.append(model_test_accs_avg)
    num_neuron_losses.append(num_neuron_loss_avg)

# Plot accuracies
final_train_acc = []
final_test_acc = []

plt.clf()
fig, ax = plt.subplots()
for idx, neuron_num in enumerate(NEURON_NUMS):
    ax.plot(range(epochs), num_neuron_test_accs[idx], label="Number of neurons = " + str(neuron_num))
    print("Neural network with {} hidden units reached train accuracy of {} and test accuracy of {} after 5000 epochs."
          .format(neuron_num, num_neuron_train_accs[idx][-1], num_neuron_test_accs[idx][-1]))
    
    final_train_acc.append(num_neuron_train_accs[idx][-1])
    final_test_acc.append(num_neuron_test_accs[idx][-1])
legend = ax.legend(loc='lower right')

plt.xlabel(str(epochs) +' iterations')
plt.ylabel('Accuracy')
plt.title('Learning Curves of 3-layer NNs with diff. neuron numbers')
plt.savefig('figures/project_1a_q3a_1.png')

plt.clf()
fig2, ax2 = plt.subplots()
ax2.plot([str(neuron_num) for neuron_num in NEURON_NUMS], final_train_acc, 'xb-', label="Final Train Accuracy")
ax2.plot([str(neuron_num) for neuron_num in NEURON_NUMS], final_test_acc, 'xr-', label="Final Test Accuracy")
legend = ax2.legend(loc='upper left')

plt.xlabel('Number of hidden neurons')
plt.ylabel('Accuracy')
plt.title('Accuracies of 3-layer NNs with diff. neuron numbers')
plt.savefig('figures/project_1a_q3a_2.png')

# Plot loss
plt.clf()
fig, ax = plt.subplots()
for idx, neuron_num in enumerate(NEURON_NUMS):
    ax.plot(range(epochs), num_neuron_losses[idx], label="Number of neurons = " + str(neuron_num))
    print("Neural network with {} hidden units has loss {} after 5000 epochs.".format(neuron_num, num_neuron_losses[idx][-1]))
legend = ax.legend(loc='upper right')

plt.xlabel(str(epochs) +' iterations')
plt.ylabel('Average epoch loss')
plt.title('Losses of 3-layer NNs with diff. neuron numbers')
plt.savefig('figures/project_1a_q3a_3.png')

# Train with optimal batch size and number of neurons
OPTIMAL_NEURON_NUM = 25

train_acc, test_acc, loss = training(cv_and_trainX, cv_and_trainY, num_classes=OPTIMAL_NEURON_NUM, decay_parameter=10 ** -6,
                                     batch_size=OPTIMAL_BATCH_SIZE, no_of_layers=3, testX=testX, testY=testY,
                                     use_small_dataset=False)

# Plot learning curves
print('Final training accuracy is {}'.format(train_acc[-1]))
print('Final test accuracy is {}'.format(test_acc[-1]))
print('Final loss is {}'.format(loss[-1]))

plt.clf()
fig, ax = plt.subplots()
ax.plot(range(epochs), train_acc, label='Training')
ax.plot(range(epochs), test_acc, label='Test')
legend = ax.legend(loc='lower right')

plt.xlabel(str(epochs) +' iterations')
plt.ylabel('Accuracy')
plt.title('Learning curve of 3-layer NN with {} hidden-layer neurons'.format(OPTIMAL_NEURON_NUM))
plt.savefig('figures/project_1a_q3c.png')

# QUESTION 4
# Optimal decay parameter

DECAY_PARAMETERS = [0, 10 ** -3, 10 ** -6, 10 ** -9, 10 ** -12]

decay_parameter_train_accs = []
decay_parameter_test_accs = []
decay_parameter_losses = []

# Try training with different batch sizes
for decay_parameter in DECAY_PARAMETERS:
    print("==============")
    print("USING DECAY PARAMETER = {}".format(decay_parameter))
    print("==============")
    train_accs = []
    cross_validation_accs = []
    losses = []
    
    # Further split trainX and trainY to do 5-fold cross-validation
    for fold in range(5):
        print("==============")
        print("FOLD {}".format(fold + 1))
        print("==============")

        # Find indexes to split dataset further.
        start_idx = fold * int(idx_split / 5)
        end_idx = (fold + 1) * int(idx_split / 5)

        # Split training dataset further to training and test set.
        trainX = np.append(cv_and_trainX[:start_idx], cv_and_trainX[end_idx:], axis=0)
        trainY = np.append(cv_and_trainY[:start_idx], cv_and_trainY[end_idx:], axis=0)
        cv_X = cv_and_trainX[start_idx:end_idx]
        cv_Y = cv_and_trainY[start_idx:end_idx]

        # Train the model.
        train_acc, cv_acc, loss = training(trainX, trainY, num_classes=OPTIMAL_NEURON_NUM, decay_parameter=decay_parameter,
                                           batch_size=OPTIMAL_BATCH_SIZE, no_of_layers=3, testX=cv_X, testY=cv_Y,
                                           use_small_dataset=False)
        
        # Record cross-validation accuracies of model
        train_accs.append(train_acc)
        cross_validation_accs.append(cv_acc)
        losses.append(loss)

    # Find mean model training accuracies per epoch and average time taken for differnt batch sizes.
    model_train_accs_avg = np.average(np.array(train_accs), axis=0).tolist()
    model_test_accs_avg = np.average(np.array(cross_validation_accs), axis=0).tolist()
    decay_parameter_loss_avg = np.average(np.array(losses), axis=0).tolist()
    
    decay_parameter_train_accs.append(model_train_accs_avg)
    decay_parameter_test_accs.append(model_test_accs_avg)
    decay_parameter_losses.append(decay_parameter_loss_avg)

# Plot accuracies
final_train_acc = []
final_test_acc = []

plt.clf()
fig, ax = plt.subplots()
for idx, decay_parameter in enumerate(DECAY_PARAMETERS):
    ax.plot(range(epochs), decay_parameter_test_accs[idx], label="Decay parameter = " + str(decay_parameter))
    print("Decay parameter = {} reached train accuracy of {} and test accuracy of {} after 5000 epochs."
          .format(decay_parameter, decay_parameter_train_accs[idx][-1], decay_parameter_test_accs[idx][-1]))
    
    final_train_acc.append(decay_parameter_train_accs[idx][-1])
    final_test_acc.append(decay_parameter_test_accs[idx][-1])
legend = ax.legend(loc='lower right')

plt.xlabel(str(epochs) +' iterations')
plt.ylabel('Accuracy')
plt.title('Learning Curves of 3-layer NNs with different decay params')
plt.savefig('figures/project_1a_q4a_1.png')

plt.clf()
fig2, ax2 = plt.subplots()
ax2.plot([str(decay_parameter) for decay_parameter in DECAY_PARAMETERS], final_train_acc, 'xb-', label="Final Train Accuracy")
ax2.plot([str(decay_parameter) for decay_parameter in DECAY_PARAMETERS], final_test_acc, 'xr-', label="Final Test Accuracy")
legend = ax2.legend(loc='upper right')

plt.xlabel('Number of hidden neurons')
plt.ylabel('Accuracy')
plt.title('Accuracies of 3-layer NNs with different decay params')
plt.savefig('figures/project_1a_q4a_2.png')

# Plot loss
plt.clf()
fig, ax = plt.subplots()
for idx, decay_parameter in enumerate(DECAY_PARAMETERS):
    print("Decay parameter = {} has loss {} after 5000 epochs.".format(decay_parameter, decay_parameter_losses[idx][-1]))
    ax.plot(range(epochs), decay_parameter_losses[idx], label="Decay parameter = " + str(decay_parameter))
legend = ax.legend(loc='center right')

plt.xlabel(str(epochs) +' iterations')
plt.ylabel('Average epoch loss')
plt.title('Losses of 3-layer NNs with different decay params')
plt.savefig('figures/project_1a_q4a_3.png')

# Train with optimal batch size and number of neurons
OPTIMAL_DECAY_PARAMETER = 10 ** -6

train_acc, test_acc, loss = training(cv_and_trainX, cv_and_trainY, num_classes=OPTIMAL_NEURON_NUM,
                                     decay_parameter=OPTIMAL_DECAY_PARAMETER, batch_size=OPTIMAL_BATCH_SIZE,
                                     no_of_layers=3, testX=testX, testY=testY, use_small_dataset=False)

# Plot learning curves
print('Final training accuracy is {}'.format(train_acc[-1]))
print('Final test accuracy is {}'.format(test_acc[-1]))
print('Final loss is {}'.format(loss[-1]))

plt.clf()
fig, ax = plt.subplots()
ax.plot(range(epochs), train_acc, label='Training')
ax.plot(range(epochs), test_acc, label='Test')
legend = ax.legend(loc='lower right')

plt.xlabel(str(epochs) +' iterations')
plt.ylabel('Accuracy')
plt.title('Learning curve of 3-layer NN with decay parameter = {}'.format(OPTIMAL_DECAY_PARAMETER))
plt.savefig('figures/project_1a_q4c.png')

# QUESTION 5
# 4-layer neuron
train_acc_4, test_acc_4, loss_4 = training(cv_and_trainX, cv_and_trainY, num_classes=10, decay_parameter=10 ** -6,
                                           batch_size=32, no_of_layers=4, testX=testX, testY=testY, use_small_dataset=False)

# Plot learning curves
plt.clf()
fig, ax = plt.subplots()
print('Final training accuracy of optimized 3-layer NN is {}'.format(train_acc[-1]))
print('Final test accuracy of optimized 3-layer NN is {}'.format(test_acc[-1]))
ax.plot(range(epochs), train_acc, label='Train (3-layer NN)')
ax.plot(range(epochs), test_acc, label='Test (3-layer NN)')
legend = ax.legend(loc='lower right')
plt.xlabel(str(epochs) +' iterations')
plt.ylabel('Accuracy')
plt.title('Learning curve of optimized 3-layer NN')
plt.savefig('figures/project_1a_q5_1.png')

plt.clf()
fig2, ax2 = plt.subplots()
print('Final training accuracy of 4-layer NN is {}'.format(train_acc_4[-1]))
print('Final test accuracy of 4-layer NN is {}'.format(test_acc_4[-1]))
ax2.plot(range(epochs), train_acc_4, label='Train (4-layer NN)')
ax2.plot(range(epochs), test_acc_4, label='Test (4-layer NN)')
legend = ax2.legend(loc='lower right')
plt.xlabel(str(epochs) +' iterations')
plt.ylabel('Accuracy')
plt.title('Learning curve of 4-layer NN')
plt.savefig('figures/project_1a_q5_2.png')

plt.clf()
fig3, ax3 = plt.subplots()
print('Final loss of 3-layer NN is {}'.format(loss[-1]))
print('Final loss of 4-layer NN is {}'.format(loss_4[-1]))
ax3.plot(range(epochs), loss, label='3-layer NN')
ax3.plot(range(epochs), loss_4, label='4-layer NN')
legend = ax3.legend(loc='upper right')
plt.xlabel('Epoch number')
plt.ylabel('Average epoch loss')
plt.title('Loss of optimized 3-layer vs 4-layer NN')
plt.savefig('figures/project_1a_q5_3.png')

plt.clf()
fig4, ax4 = plt.subplots()
print('Final test accuracy of optimized 3-layer NN is {}'.format(test_acc[-1]))
print('Final test accuracy of 4-layer NN is {}'.format(test_acc_4[-1]))
ax4.plot(range(epochs), test_acc, label='3-layer NN')
ax4.plot(range(epochs), test_acc_4, label='4-layer NN')
legend = ax4.legend(loc='lower right')
plt.xlabel('Epoch number')
plt.ylabel('Test Accuracy')
plt.title('Test Accuracy of optimized 3-layer vs 4-layer NN')
plt.savefig('figures/project_1a_q5_4.png')

# Train with optimzed hyperparameters
train_acc_4_opt, test_acc_4_opt, loss_4_opt = training(cv_and_trainX, cv_and_trainY, num_classes=OPTIMAL_NEURON_NUM,
                                                       decay_parameter=OPTIMAL_DECAY_PARAMETER,batch_size=OPTIMAL_BATCH_SIZE,
                                                       no_of_layers=4, testX=testX, testY=testY, use_small_dataset=False)

plt.clf()
fig, ax = plt.subplots()
print('Final training accuracy of 4-layer NN is {}'.format(train_acc_4_opt[-1]))
print('Final test accuracy of 4-layer NN is {}'.format(test_acc_4_opt[-1]))
ax.plot(range(epochs), train_acc_4_opt, label='Train (Opt. 4-layer NN)')
ax.plot(range(epochs), test_acc_4_opt, label='Test (Opt. 4-layer NN)')
legend = ax.legend(loc='lower right')
plt.xlabel(str(epochs) +' iterations')
plt.ylabel('Accuracy')
plt.title('Learning curve of optimized 4-layer NN')
plt.savefig('figures/project_1a_q5_5.png')

plt.clf()
fig2, ax2 = plt.subplots()
print('Final loss of optimized 3-layer NN is {}'.format(loss[-1]))
print('Final loss of optimized 4-layer NN is {}'.format(loss_4_opt[-1]))
ax2.plot(range(epochs), loss, label='Opt. 3-layer NN')
ax2.plot(range(epochs), loss_4_opt, label='Opt. 4-layer NN')
legend = ax2.legend(loc='upper right')
plt.xlabel('Epoch number')
plt.ylabel('Average epoch loss')
plt.title('Loss of optimized 3-layer vs optimized 4-layer NN')
plt.savefig('figures/project_1a_q5_6.png')

plt.clf()
fig3, ax3 = plt.subplots()
print('Final test accuracy of optimized 3-layer NN is {}'.format(test_acc[-1]))
print('Final test accuracy of optimized 4-layer NN is {}'.format(test_acc_4_opt[-1]))
ax3.plot(range(epochs), test_acc, label='Opt. 3-layer NN')
ax3.plot(range(epochs), test_acc_4_opt, label='Opt. 4-layer NN')
legend = ax3.legend(loc='lower right')
plt.xlabel('Epoch number')
plt.ylabel('Test Accuracy')
plt.title('Test Accuracy of optimized 3-layer vs optimized 4-layer NN')
plt.savefig('figures/project_1a_q5_7.png')