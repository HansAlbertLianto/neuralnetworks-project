import pandas as pd
import numpy as np
import tensorflow as tf

from matplotlib import pyplot as plt

MAX_INPUT_LENGTH = 100
MAX_NO_OF_CHARACTERS = 256

N_CONV_FILTERS = 10
EMBEDDING_SIZE = 20
HIDDEN_LAYER_SIZE = 20
MAX_LABEL = 15
BATCH_SIZE = 128

GRADIENT_CLIP_VALUE = 0.125
DROPOUT_RATE = 0.2

NO_OF_EPOCHS = 100
LEARNING_RATE = 0.01

tf.logging.set_verbosity(tf.logging.ERROR)
seed = 10
tf.set_random_seed(seed)

# Function to read data from CSV file
def read_data(encoding_level):
  # Get Pandas dataframe
  raw_data_train = pd.read_csv('train_medium.csv', names=["Category", "Wikipedia Page", "Paragraph Text"])
  raw_data_test = pd.read_csv('test_medium.csv', names=["Category", "Wikipedia Page", "Paragraph Text"])

  # Slice Pandas dataframe to get input and output attribute
  if encoding_level == "char":
    input_text_train, input_text_test = pd.Series(raw_data_train["Wikipedia Page"]), pd.Series(raw_data_test["Wikipedia Page"])
  elif encoding_level == "word":
    input_text_train, input_text_test = pd.Series(raw_data_train["Paragraph Text"]), pd.Series(raw_data_test["Paragraph Text"])
  output_label_train, output_label_test = pd.Series(raw_data_train["Category"]), pd.Series(raw_data_test["Category"])

  # Encode character to text
  if encoding_level == "char":
    encoder = tf.contrib.learn.preprocessing.ByteProcessor(MAX_INPUT_LENGTH)
  elif encoding_level == "word":
    encoder = tf.contrib.learn.preprocessing.VocabularyProcessor(MAX_INPUT_LENGTH)

  x_train = np.array(list(encoder.fit_transform(input_text_train)), dtype=np.uint8)
  y_train = np.array(output_label_train, dtype=np.uint8).reshape((-1, 1))

  x_test = np.array(list(encoder.fit_transform(input_text_test)), dtype=np.uint8)
  y_test = np.array(output_label_test, dtype=np.uint8).reshape((-1, 1))

  if encoding_level == "word":
    no_of_words = len(encoder.vocabulary_)
  else:
    no_of_words = 0

  return x_train, y_train, x_test, y_test, no_of_words


# RNN Model
def rnn_model(x, encoding_level, no_of_words=None, no_of_layers=1, cell_used="GRU", use_dropout=False):
  dropout_rate = tf.placeholder(tf.float32)
  
  if encoding_level == "char":
    char_vectors = tf.one_hot(x, MAX_NO_OF_CHARACTERS)
    encoding_list = tf.unstack(char_vectors, axis=1)
  elif encoding_level == "word":
    word_vectors = tf.contrib.layers.embed_sequence(
        x, vocab_size=no_of_words, embed_dim=EMBEDDING_SIZE)
    encoding_list = tf.unstack(word_vectors, axis=1)

  # Multi-layer RNN
  if no_of_layers > 1:

    # Fill with each layer with RNN cell
    cell_layers = []
    for i in range(no_of_layers):
      if cell_used == "GRU":
        cell_layers.append(tf.nn.rnn_cell.GRUCell(HIDDEN_LAYER_SIZE))
      elif cell_used == "Vanilla":
        cell_layers.append(tf.nn.rnn_cell.BasicRNNCell(HIDDEN_LAYER_SIZE))
      elif cell_used == "LSTM":
        cell_layers.append(tf.nn.rnn_cell.LSTMCell(HIDDEN_LAYER_SIZE))

    # Create multilayer RNN cell
    cell = tf.nn.rnn_cell.MultiRNNCell(cell_layers)

  elif no_of_layers == 1:
    if cell_used == "GRU":
      cell = tf.nn.rnn_cell.GRUCell(HIDDEN_LAYER_SIZE)
    elif cell_used == "Vanilla":
      cell = tf.nn.rnn_cell.BasicRNNCell(HIDDEN_LAYER_SIZE)
    elif cell_used == "LSTM":
      cell = tf.nn.rnn_cell.LSTMCell(HIDDEN_LAYER_SIZE)

  # Dropout
  if use_dropout:
    cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=1-dropout_rate,
                                        output_keep_prob=1-dropout_rate)
    
  _, states = tf.nn.static_rnn(cell, encoding_list, dtype=tf.float32)

  # Get only the hidden-layer activations (h), not state (c)
  if isinstance(states, tuple):
    states = states[-1]

  # Dense layer from hidden-layer activations to output
  logits = tf.layers.dense(states, MAX_LABEL, activation=None)

  return logits, dropout_rate


def plot_graph(title, fig_title, question_no, data_list, y_label, legend_labels):
  fig, ax = plt.subplots()
  for data, label in zip(data_list, legend_labels):
    ax.plot(list(range(1, NO_OF_EPOCHS + 1)), data, label=label)
  legend = ax.legend(loc='best')

  plt.xlabel(str(NO_OF_EPOCHS) +' iterations')
  plt.ylabel(y_label)
  plt.title(title)
  plt.savefig('figures/project_2b_q{}_{}'.format(question_no, fig_title))


def training(encoding_level, cell_used="GRU", no_of_layers=1, use_dropout=False, use_gradient_clip=False, use_timer=False):
  # Read and encode data
  x_train, y_train, x_test, y_test, no_of_words = read_data(encoding_level)

  # Reseed so shuffled data for mini-batch gradient descent are consistent with each training
  np.random.seed(seed)

  # Create the model
  x = tf.placeholder(tf.int64, [None, MAX_INPUT_LENGTH])
  y_ = tf.placeholder(tf.int64, [None, 1])

  logits, dropout_rate = rnn_model(x, encoding_level, no_of_words=no_of_words, no_of_layers=no_of_layers, cell_used=cell_used, use_dropout=use_dropout)

  # Optimizer
  entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y_, MAX_LABEL), logits=logits))
  
  if use_gradient_clip:
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
    gradients_and_values = optimizer.compute_gradients(entropy)

    # Clip gradients, then apply during mini-batch gradient descent.
    clipped_gradients_and_values = [(tf.clip_by_value(gradient, -GRADIENT_CLIP_VALUE, GRADIENT_CLIP_VALUE), value) for gradient, value in gradients_and_values]
    train_op = optimizer.apply_gradients(clipped_gradients_and_values)
  else:
    train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(entropy)

  # Accuracy
  accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.reshape(tf.argmax(logits, 1), [-1, 1]), y_), tf.float32))

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # training
    train_losses = []
    train_accuracies = []

    test_losses = []
    test_accuracies = []

    # Start timer
    if use_timer:
      start_time = time.time()

    for e in range(NO_OF_EPOCHS):
      # Shuffle the dataset before grouping them into minibatches for gradient updates
      dataset_size = x_train.shape[0]
      idx = np.arange(dataset_size)
      np.random.shuffle(idx)
      x_shuffled, y_shuffled = x_train[idx], y_train[idx]
      
      for start, end in zip(range(0, len(x_train), BATCH_SIZE), range(BATCH_SIZE, len(x_train), BATCH_SIZE)):
        _ = sess.run(train_op, feed_dict={x: x_shuffled[start:end], y_: y_shuffled[start:end], dropout_rate: DROPOUT_RATE})
      
      train_loss_, train_accuracy_ = sess.run([entropy, accuracy], feed_dict={x: x_train, y_: y_train, dropout_rate: 0})
      train_losses.append(train_loss_)
      train_accuracies.append(train_accuracy_)

      test_loss_, test_accuracy_ = sess.run([entropy, accuracy], feed_dict={x: x_test, y_: y_test, dropout_rate: 0})
      test_losses.append(test_loss_)
      test_accuracies.append(test_accuracy_)

      if e % 1 == 0:
        print('iter: %d, train_entropy: %g, train_accuracy: %g, test_entropy: %g, test_accuracy: %g' % (e, train_losses[e], train_accuracies[e], test_losses[e], test_accuracies[e]))
    
    if use_timer:
      end_time = time.time() - start_time
    else:
      end_time = 0

  return train_losses, train_accuracies, test_losses, test_accuracies, end_time


def main():
  # Train Character RNN model
  entropy_losses_char_rnn, train_accuracies_char_rnn, test_losses_char_rnn, test_accuracies_char_rnn, end_time_char = training(encoding_level='char', use_timer=True)

  print("=========== Char-RNN ===========")
  print("Final entropy loss: {}".format(entropy_losses_char_rnn[-1]))
  print("Final train accuracy: {}".format(train_accuracies_char_rnn[-1]))
  print("Final test loss: {}".format(test_losses_char_rnn[-1]))
  print("Final test accuracy: {}".format(test_accuracies_char_rnn[-1]))
  print("Time taken to train: {} seconds".format(end_time_char))
  print()

  plot_graph(title="Character-RNN Model Learning Curve", 
             fig_title="char_rnn_learning_curve", 
             question_no="3", 
             data_list=[entropy_losses_char_rnn, test_losses_char_rnn],
             y_label="Entropy Loss",
             legend_labels=["Train", "Test"])
  plot_graph(title="Character-RNN Model Accuracy vs Epochs", 
             fig_title="char_rnn_accuracy", 
             question_no="3", 
             data_list=[train_accuracies_char_rnn, test_accuracies_char_rnn], 
             y_label="Accuracy", 
             legend_labels=["Train", "Test"])
  
  tf.reset_default_graph()

  # Train Word RNN model
  entropy_losses_word_rnn, train_accuracies_word_rnn, test_losses_word_rnn, test_accuracies_word_rnn, end_time_word = training(encoding_level='word', use_timer=True)

  print("=========== Word-RNN ===========")
  print("Final entropy loss: {}".format(entropy_losses_word_rnn[-1]))
  print("Final train accuracy: {}".format(train_accuracies_word_rnn[-1]))
  print("Final test loss: {}".format(test_losses_word_rnn[-1]))
  print("Final test accuracy: {}".format(test_accuracies_word_rnn[-1]))
  print("Time taken to train: {} seconds".format(end_time_word))
  print()

  plot_graph(title="Word-RNN Model Learning Curve", 
             fig_title="word_rnn_learning_curve", 
             question_no="4", 
             data_list=[entropy_losses_word_rnn, test_losses_word_rnn],
             y_label="Entropy Loss",
             legend_labels=["Train", "Test"])
  plot_graph(title="Word-RNN Model Accuracy vs Epochs", 
             fig_title="word_rnn_accuracy", 
             question_no="4", 
             data_list=[train_accuracies_word_rnn, test_accuracies_word_rnn], 
             y_label="Accuracy", 
             legend_labels=["Train", "Test"])
  
  tf.reset_default_graph()

  plt.clf()

  # Plot histogram
  models = ['Char-RNN', 'Word-RNN']
  y_pos = np.arange(len(models))
  times = [end_time_char, end_time_word]

  plt.bar(models, times, align='center', alpha=0.5)
  plt.xticks(y_pos, models)
  plt.ylabel('Time taken to train')
  plt.title('Time taken to train vs model')
  plt.savefig('figures/project_2b_q5_rnn_time.png')

  plt.clf()
  
  # Train Character RNN model with Dropout
  entropy_losses_char_rnn_dropout, train_accuracies_char_rnn_dropout, test_losses_char_rnn_dropout, test_accuracies_char_rnn_dropout, _ = training(encoding_level='char', use_dropout=True)

  print("=========== Char-RNN with Dropout ===========")
  print("Final entropy loss: {}".format(entropy_losses_char_rnn_dropout[-1]))
  print("Final train accuracy: {}".format(train_accuracies_char_rnn_dropout[-1]))
  print("Final test loss: {}".format(test_losses_char_rnn_dropout[-1]))
  print("Final test accuracy: {}".format(test_accuracies_char_rnn_dropout[-1]))
  print()

  plot_graph(title="Character-RNN Model Learning Curve w/ Dropout", 
             fig_title="char_rnn_learning_curve_dropout", 
             question_no="5", 
             data_list=[entropy_losses_char_rnn_dropout, test_losses_char_rnn_dropout],
             y_label="Entropy Loss",
             legend_labels=["Train", "Test"])
  plot_graph(title="Character-RNN Model Accuracy vs Epochs w/ Dropout", 
             fig_title="char_rnn_accuracy_dropout", 
             question_no="5", 
             data_list=[train_accuracies_char_rnn_dropout, test_accuracies_char_rnn_dropout], 
             y_label="Accuracy", 
             legend_labels=["Train", "Test"])
  
  tf.reset_default_graph()
  
  # Train Word RNN model with Dropout
  entropy_losses_word_rnn_dropout, train_accuracies_word_rnn_dropout, test_losses_word_rnn_dropout, test_accuracies_word_rnn_dropout, _ = training(encoding_level='word', use_dropout=True)

  print("=========== Word-RNN with Dropout ===========")
  print("Final entropy loss: {}".format(entropy_losses_word_rnn_dropout[-1]))
  print("Final train accuracy: {}".format(train_accuracies_word_rnn_dropout[-1]))
  print("Final test loss: {}".format(test_losses_word_rnn_dropout[-1]))
  print("Final test accuracy: {}".format(test_accuracies_word_rnn_dropout[-1]))
  print()

  plot_graph(title="Word-RNN Model Learning Curve w/ Dropout", 
             fig_title="word_rnn_learning_curve_dropout", 
             question_no="5", 
             data_list=[entropy_losses_word_rnn_dropout, test_losses_word_rnn_dropout],
             y_label="Entropy Loss",
             legend_labels=["Train", "Test"])
  plot_graph(title="Word-RNN Model Accuracy vs Epochs w/ Dropout", 
             fig_title="word_rnn_accuracy_dropout", 
             question_no="5", 
             data_list=[train_accuracies_word_rnn_dropout, test_accuracies_word_rnn_dropout], 
             y_label="Accuracy", 
             legend_labels=["Train", "Test"])
  
  tf.reset_default_graph()
  
  # Train Character RNN model with Vanilla cell
  entropy_losses_char_rnn_vanilla, train_accuracies_char_rnn_vanilla, test_losses_char_rnn_vanilla, test_accuracies_char_rnn_vanilla, _ = training(encoding_level='char', cell_used="Vanilla")

  print("=========== Char Vanilla-cell RNN ===========")
  print("Final entropy loss: {}".format(entropy_losses_char_rnn_vanilla[-1]))
  print("Final train accuracy: {}".format(train_accuracies_char_rnn_vanilla[-1]))
  print("Final test loss: {}".format(test_losses_char_rnn_vanilla[-1]))
  print("Final test accuracy: {}".format(test_accuracies_char_rnn_vanilla[-1]))
  print()

  plot_graph(title="Character Vanilla-RNN Model Learning Curve", 
             fig_title="char_rnn_learning_curve_vanilla", 
             question_no="6", 
             data_list=[entropy_losses_char_rnn_vanilla, test_losses_char_rnn_vanilla],
             y_label="Entropy Loss",
             legend_labels=["Train", "Test"])
  plot_graph(title="Character Vanilla-RNN Model Accuracy vs Epochs", 
             fig_title="char_rnn_accuracy_vanilla", 
             question_no="6", 
             data_list=[train_accuracies_char_rnn_vanilla, test_accuracies_char_rnn_vanilla], 
             y_label="Accuracy", 
             legend_labels=["Train", "Test"])
  
  tf.reset_default_graph()
  
  # Train Character RNN model with LSTM cell
  entropy_losses_char_rnn_lstm, train_accuracies_char_rnn_lstm, test_losses_char_rnn_lstm, test_accuracies_char_rnn_lstm, _ = training(encoding_level='char', cell_used="LSTM")

  print("=========== Char LSTM-cell RNN ===========")
  print("Final entropy loss: {}".format(entropy_losses_char_rnn_lstm[-1]))
  print("Final train accuracy: {}".format(train_accuracies_char_rnn_lstm[-1]))
  print("Final test loss: {}".format(test_losses_char_rnn_lstm[-1]))
  print("Final test accuracy: {}".format(test_accuracies_char_rnn_lstm[-1]))
  print()

  plot_graph(title="Character LSTM-cell RNN Model Learning Curve", 
             fig_title="char_rnn_learning_curve_lstm", 
             question_no="6", 
             data_list=[entropy_losses_char_rnn_lstm, test_losses_char_rnn_lstm],
             y_label="Entropy Loss",
             legend_labels=["Train", "Test"])
  plot_graph(title="Character LSTM-cell RNN Model Accuracy vs Epochs", 
             fig_title="char_rnn_accuracy_lstm", 
             question_no="6", 
             data_list=[train_accuracies_char_rnn_lstm, test_accuracies_char_rnn_lstm], 
             y_label="Accuracy", 
             legend_labels=["Train", "Test"])
  
  tf.reset_default_graph()
  
  # Train 2-layer Character RNN model
  entropy_losses_char_rnn_2_layer, train_accuracies_char_rnn_2_layer, test_losses_char_rnn_2_layer, test_accuracies_char_rnn_2_layer, _ = training(encoding_level='char', no_of_layers=2)

  print("=========== 2-layer Char RNN ===========")
  print("Final entropy loss: {}".format(entropy_losses_char_rnn_2_layer[-1]))
  print("Final train accuracy: {}".format(train_accuracies_char_rnn_2_layer[-1]))
  print("Final test loss: {}".format(test_losses_char_rnn_2_layer[-1]))
  print("Final test accuracy: {}".format(test_accuracies_char_rnn_2_layer[-1]))
  print()

  plot_graph(title="2-layer Character RNN Model Learning Curve", 
             fig_title="char_rnn_learning_curve_for_2_layers", 
             question_no="6", 
             data_list=[entropy_losses_char_rnn_2_layer, test_losses_char_rnn_2_layer],
             y_label="Entropy Loss",
             legend_labels=["Train", "Test"])
  plot_graph(title="2-layer Character RNN Model Accuracy vs Epochs", 
             fig_title="char_rnn_accuracy_for_2_layers", 
             question_no="6", 
             data_list=[train_accuracies_char_rnn_2_layer, test_accuracies_char_rnn_2_layer], 
             y_label="Accuracy", 
             legend_labels=["Train", "Test"])
  
  tf.reset_default_graph()
  
  # Train Character RNN model with gradient clipping
  entropy_losses_char_rnn_grad_clip, train_accuracies_char_rnn_grad_clip, test_losses_char_rnn_grad_clip, test_accuracies_char_rnn_grad_clip, _ = training(encoding_level='char', use_gradient_clip=True)

  print("=========== Char RNN with Gradient Clipping ===========")
  print("Final entropy loss: {}".format(entropy_losses_char_rnn_grad_clip[-1]))
  print("Final train accuracy: {}".format(train_accuracies_char_rnn_grad_clip[-1]))
  print("Final test loss: {}".format(test_losses_char_rnn_grad_clip[-1]))
  print("Final test accuracy: {}".format(test_accuracies_char_rnn_grad_clip[-1]))
  print()

  plot_graph(title="Character RNN Model with Gradient Clipping Learning Curve", 
             fig_title="char_rnn_learning_curve_gradient_clipping", 
             question_no="6", 
             data_list=[entropy_losses_char_rnn_grad_clip, test_losses_char_rnn_grad_clip],
             y_label="Entropy Loss",
             legend_labels=["Train", "Test"])
  plot_graph(title="Character RNN Model with Gradient Clipping Accuracy vs Epochs", 
             fig_title="char_rnn_accuracy_gradient_clipping", 
             question_no="6", 
             data_list=[train_accuracies_char_rnn_grad_clip, test_accuracies_char_rnn_grad_clip], 
             y_label="Accuracy", 
             legend_labels=["Train", "Test"])
  
  tf.reset_default_graph()
  
  # Train Word RNN model with Vanilla cell
  entropy_losses_word_rnn_vanilla, train_accuracies_word_rnn_vanilla, test_losses_word_rnn_vanilla, test_accuracies_word_rnn_vanilla, _ = training(encoding_level='word', cell_used="Vanilla")

  print("=========== Word Vanilla-cell RNN ===========")
  print("Final entropy loss: {}".format(entropy_losses_word_rnn_vanilla[-1]))
  print("Final train accuracy: {}".format(train_accuracies_word_rnn_vanilla[-1]))
  print("Final test loss: {}".format(test_losses_word_rnn_vanilla[-1]))
  print("Final test accuracy: {}".format(test_accuracies_word_rnn_vanilla[-1]))
  print()

  plot_graph(title="Word Vanilla-RNN Model Learning Curve", 
             fig_title="word_rnn_learning_curve_vanilla", 
             question_no="6", 
             data_list=[entropy_losses_word_rnn_vanilla, test_losses_word_rnn_vanilla],
             y_label="Entropy Loss",
             legend_labels=["Train", "Test"])
  plot_graph(title="Word Vanilla-RNN Model Accuracy vs Epochs", 
             fig_title="word_rnn_accuracy_vanilla", 
             question_no="6", 
             data_list=[train_accuracies_word_rnn_vanilla, test_accuracies_word_rnn_vanilla], 
             y_label="Accuracy", 
             legend_labels=["Train", "Test"])
  
  tf.reset_default_graph()
  
  # Train Word RNN model with LSTM cell
  entropy_losses_word_rnn_lstm, train_accuracies_word_rnn_lstm, test_losses_word_rnn_lstm, test_accuracies_word_rnn_lstm, _ = training(encoding_level='word', cell_used="LSTM")

  print("=========== Word LSTM-cell RNN ===========")
  print("Final entropy loss: {}".format(entropy_losses_word_rnn_lstm[-1]))
  print("Final train accuracy: {}".format(train_accuracies_word_rnn_lstm[-1]))
  print("Final test loss: {}".format(test_losses_word_rnn_lstm[-1]))
  print("Final test accuracy: {}".format(test_accuracies_word_rnn_lstm[-1]))
  print()

  plot_graph(title="Word LSTM-cell RNN Model Learning Curve", 
             fig_title="word_rnn_learning_curve_lstm", 
             question_no="6", 
             data_list=[entropy_losses_word_rnn_lstm, test_losses_word_rnn_lstm],
             y_label="Entropy Loss",
             legend_labels=["Train", "Test"])
  plot_graph(title="Word LSTM-cell RNN Model Accuracy vs Epochs", 
             fig_title="word_rnn_accuracy_lstm", 
             question_no="6", 
             data_list=[train_accuracies_word_rnn_lstm, test_accuracies_word_rnn_lstm], 
             y_label="Accuracy", 
             legend_labels=["Train", "Test"])
  
  tf.reset_default_graph()
  
  # Train 2-layer Word RNN model
  entropy_losses_word_rnn_2_layer, train_accuracies_word_rnn_2_layer, test_losses_word_rnn_2_layer, test_accuracies_word_rnn_2_layer, _ = training(encoding_level='word', no_of_layers=2)

  print("=========== 2-layer Word RNN ===========")
  print("Final entropy loss: {}".format(entropy_losses_word_rnn_2_layer[-1]))
  print("Final train accuracy: {}".format(train_accuracies_word_rnn_2_layer[-1]))
  print("Final test loss: {}".format(test_losses_word_rnn_2_layer[-1]))
  print("Final test accuracy: {}".format(test_accuracies_word_rnn_2_layer[-1]))
  print()

  plot_graph(title="2-layer Word RNN Model Learning Curve", 
             fig_title="word_rnn_learning_curve_for_2_layers", 
             question_no="6", 
             data_list=[entropy_losses_word_rnn_2_layer, test_losses_word_rnn_2_layer],
             y_label="Entropy Loss",
             legend_labels=["Train", "Test"])
  plot_graph(title="2-layer Word RNN Model Accuracy vs Epochs", 
             fig_title="word_rnn_accuracy_for_2_layers", 
             question_no="6", 
             data_list=[train_accuracies_word_rnn_2_layer, test_accuracies_word_rnn_2_layer], 
             y_label="Accuracy", 
             legend_labels=["Train", "Test"])
  
  tf.reset_default_graph()
  
  # Train Word RNN model with gradient clipping
  entropy_losses_word_rnn_grad_clip, train_accuracies_word_rnn_grad_clip, test_losses_word_rnn_grad_clip, test_accuracies_word_rnn_grad_clip, _ = training(encoding_level='word', use_gradient_clip=True)

  print("=========== Word RNN with Gradient Clipping ===========")
  print("Final entropy loss: {}".format(entropy_losses_word_rnn_grad_clip[-1]))
  print("Final train accuracy: {}".format(train_accuracies_word_rnn_grad_clip[-1]))
  print("Final test loss: {}".format(test_losses_word_rnn_grad_clip[-1]))
  print("Final test accuracy: {}".format(test_accuracies_word_rnn_grad_clip[-1]))
  print()

  plot_graph(title="Word RNN Model with Gradient Clipping Learning Curve", 
             fig_title="word_rnn_learning_curve_gradient_clipping", 
             question_no="6", 
             data_list=[entropy_losses_word_rnn_grad_clip, test_losses_word_rnn_grad_clip],
             y_label="Entropy Loss",
             legend_labels=["Train", "Test"])
  plot_graph(title="Word RNN Model with Gradient Clipping Accuracy vs Epochs", 
             fig_title="word_rnn_accuracy_gradient_clipping", 
             question_no="6", 
             data_list=[train_accuracies_word_rnn_grad_clip, test_accuracies_word_rnn_grad_clip], 
             y_label="Accuracy", 
             legend_labels=["Train", "Test"])
  
  # Comparison of different methods
  print("Plotting more comparison graphs...")

  plot_graph(title="Char RNN Model with and without Dropout (Loss)", 
             fig_title="char_rnn_learning_curve_with_and_without_dropout", 
             question_no="5", 
             data_list=[entropy_losses_char_rnn,
                        entropy_losses_char_rnn_dropout],
             y_label="Loss",
             legend_labels=["GRU", "GRU w/ Dropout (0.8)"])
  
  plot_graph(title="Char RNN Model with and without Dropout (Accuracy)", 
             fig_title="char_rnn_accuracy_with_and_without_dropout", 
             question_no="5", 
             data_list=[test_accuracies_char_rnn,
                        test_accuracies_char_rnn_dropout],
             y_label="Test Accuracy",
             legend_labels=["GRU", "GRU w/ Dropout (0.8)"])
  
  plot_graph(title="Word RNN Model with and without Dropout (Loss)", 
             fig_title="word_rnn_learning_curve_with_and_without_dropout", 
             question_no="5", 
             data_list=[entropy_losses_word_rnn,
                        entropy_losses_word_rnn_dropout],
             y_label="Loss",
             legend_labels=["GRU", "GRU w/ Dropout (0.8)"])
  
  plot_graph(title="Word RNN Model with and without Dropout (Accuracy)", 
             fig_title="word_rnn_accuracy_with_and_without_dropout", 
             question_no="5", 
             data_list=[test_accuracies_word_rnn,
                        test_accuracies_word_rnn_dropout],
             y_label="Test Accuracy",
             legend_labels=["GRU", "GRU w/ Dropout (0.8)"])
  
  plot_graph(title="Char RNN Model with Different Cells (Loss)", 
             fig_title="char_rnn_learning_curve_with_different_cells", 
             question_no="6", 
             data_list=[entropy_losses_char_rnn,
                        entropy_losses_char_rnn_vanilla,
                        entropy_losses_char_rnn_lstm],
             y_label="Loss",
             legend_labels=["GRU", "Vanilla (BasicRNNCell)", "LSTM"])
  
  plot_graph(title="Char RNN Model with Different Cells (Accuracy)", 
             fig_title="char_rnn_accuracy_with_different_cells", 
             question_no="6", 
             data_list=[test_accuracies_char_rnn,
                        test_accuracies_char_rnn_vanilla,
                        test_accuracies_char_rnn_lstm],
             y_label="Test Accuracy",
             legend_labels=["GRU", "Vanilla (BasicRNNCell)", "LSTM"])
  
  plot_graph(title="Word RNN Model with Different Cells (Loss)", 
             fig_title="word_rnn_learning_curve_with_different_cells", 
             question_no="6", 
             data_list=[entropy_losses_word_rnn,
                        entropy_losses_word_rnn_vanilla,
                        entropy_losses_word_rnn_lstm],
             y_label="Loss",
             legend_labels=["GRU", "Vanilla (BasicRNNCell)", "LSTM"])
  
  plot_graph(title="Word RNN Model with Different Cells (Accuracy)", 
             fig_title="word_rnn_accuracy_with_different_cells", 
             question_no="6", 
             data_list=[test_accuracies_word_rnn,
                        test_accuracies_word_rnn_vanilla,
                        test_accuracies_word_rnn_lstm],
             y_label="Test Accuracy",
             legend_labels=["GRU", "Vanilla (BasicRNNCell)", "LSTM"])
  
  plot_graph(title="Char RNN Model with 1 layer vs 2 layers (Loss)", 
             fig_title="char_rnn_learning_curve_with_1_and_2_layers", 
             question_no="6", 
             data_list=[entropy_losses_char_rnn,
                        entropy_losses_char_rnn_2_layer],
             y_label="Loss",
             legend_labels=["GRU (1 layer)", "GRU (2 layers)"])
  
  plot_graph(title="Char RNN Model with 1 layer vs 2 layers (Accuracy)", 
             fig_title="char_rnn_accuracy_with_1_and_2_layers", 
             question_no="6", 
             data_list=[test_accuracies_char_rnn,
                        test_accuracies_char_rnn_2_layer],
             y_label="Test Accuracy",
             legend_labels=["GRU (1 layer)", "GRU (2 layers)"])
  
  plot_graph(title="Word RNN Model with 1 layer vs 2 layers (Loss)", 
             fig_title="word_rnn_learning_curve_with_1_and_2_layers", 
             question_no="6", 
             data_list=[entropy_losses_word_rnn,
                        entropy_losses_word_rnn_2_layer],
             y_label="Loss",
             legend_labels=["GRU (1 layer)", "GRU (2 layers)"])
  
  plot_graph(title="Word RNN Model with 1 layer vs 2 layers (Accuracy)", 
             fig_title="word_rnn_accuracy_with_1_and_2_layers", 
             question_no="6", 
             data_list=[test_accuracies_word_rnn,
                        test_accuracies_word_rnn_2_layer],
             y_label="Test Accuracy",
             legend_labels=["GRU (1 layer)", "GRU (2 layers)"])
  
  plot_graph(title="Char RNN Model with and without Gradient Clipping (Loss)", 
             fig_title="char_rnn_learning_curve_with_and_without_grad_clip", 
             question_no="6", 
             data_list=[entropy_losses_char_rnn,
                        entropy_losses_char_rnn_grad_clip],
             y_label="Loss",
             legend_labels=["GRU (w/out Grad Clip)", "GRU (w/ Grad Clip)"])
  
  plot_graph(title="Char RNN Model with and without Gradient Clipping (Accuracy)", 
             fig_title="char_rnn_accuracy_with_and_without_grad_clip", 
             question_no="6", 
             data_list=[test_accuracies_char_rnn,
                        test_accuracies_char_rnn_grad_clip],
             y_label="Test Accuracy",
             legend_labels=["GRU (w/out Grad Clip)", "GRU (w/ Grad Clip)"])
  
  plot_graph(title="Word RNN Model with and without Gradient Clipping (Loss)", 
             fig_title="word_rnn_learning_curve_with_and_without_grad_clip", 
             question_no="6", 
             data_list=[entropy_losses_word_rnn,
                        entropy_losses_word_rnn_grad_clip],
             y_label="Loss",
             legend_labels=["GRU (w/out Grad Clip)", "GRU (w/ Grad Clip)"])
  
  plot_graph(title="Word RNN Model with and without Gradient Clipping (Accuracy)", 
             fig_title="word_rnn_accuracy_with_and_without_grad_clip", 
             question_no="6", 
             data_list=[test_accuracies_word_rnn,
                        test_accuracies_word_rnn_grad_clip],
             y_label="Test Accuracy",
             legend_labels=["GRU (w/out Grad Clip)", "GRU (w/ Grad Clip)"])


if __name__ == "__main__":
  main()