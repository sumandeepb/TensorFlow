# Train Neural Network model to classify movie review texts
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

print("Using TensorFlow: ", tf.__version__)

# 50K movie reviews
imdb = keras.datasets.imdb

# 25K training set, 25K testing set
# word dictionary limited to 10K most frequently occuring words only
(train_data, train_labels), (test_data,
                             test_labels) = imdb.load_data(num_words=10000)

# count training samples
print("Training entries: {}, labels: {}".format(
    len(train_data), len(train_labels)))

# word to index dictionary mapping
word_index = imdb.get_word_index()

# add reserved word to first indices
word_index = {k: (v+3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

# create index to word mapping
reverse_word_index = dict([value, key] for (key, value) in word_index.items())


# decoder function to get original text review
def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

# print(len(train_data[0]))
# print(train_data[0])
# print(decode_review(train_data[0]))


# pad the data to create all sequences of size max_length
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)

# print(len(train_data[0]))
# print(decode_review(train_data[0]))

# input shape is the vocabulary count used for the movie reviews
vocab_size = 10000

# define the model architecture
model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
model.summary()

# compile the model with appropriate optimizer and loss function
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# create validation set of 10K samples
x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

# train the model
history = model.fit(partial_x_train, partial_y_train,
                    epochs=40, batch_size=512,
                    validation_data=(x_val, y_val), verbose=1)

# evaluate test results on the model
results = model.evaluate(test_data, test_labels)
print(results)

# plot training convergence curves
history_dict = history.history
history_dict.keys()

# fetch training data from history
loss = history_dict['loss']
val_loss = history_dict['val_loss']
acc = history_dict['acc']
val_acc = history_dict['val_acc']

epochs = range(1, len(acc) + 1)

# plot loss graph
p1 = plt.figure(1)
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
p1.show()

# plot accuracy graph
p2 = plt.figure(2)
plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
p2.show()

# holdup plots till input
input()
