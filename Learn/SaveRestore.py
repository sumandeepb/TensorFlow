# demostrate save and load capabilities of keras
from __future__ import absolute_import, division, print_function
import os
import tensorflow as tf
from tensorflow import keras

print("Using TensorFlow: ", tf.__version__)

(train_images, train_labels), (test_images,
                               test_labels) = keras.datasets.mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0


# define model architecture
def create_model():
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(512, activation=tf.nn.relu),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# create checkpoint callback
checkpoint_path = './cp-{epoch:04d}.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = keras.callbacks.ModelCheckpoint(checkpoint_path, verbose=1,
                                              save_weights_only=True,
                                              period=5)

# create model instance
model = create_model()
model.summary()

# train model
model.fit(train_images, train_labels, epochs=20,
          validation_data=(test_images, test_labels),
          callbacks=[cp_callback])
model.save_weights('./my_checkpoint.ckpt')
model.save('./my_model.h5')
# loaded using keras.models.load_model('./my_model.h5')

# create new untrained model
model = create_model()
loss, acc = model.evaluate(test_images, test_labels)
print("Untrained model accuracy: {:5.2f}%".format(100*acc))

# get latest checkpoint
latest = tf.train.latest_checkpoint(checkpoint_dir)
print("Latest checkpoint: ", latest)

# load model and re-evaluate
model.load_weights(latest)
loss, acc = model.evaluate(test_images, test_labels)
print("Restored model accuracy: {:5.2f}%".format(100*acc))

# alternative save and load
saved_model_path = tf.contrib.saved_model.save_keras_model(
    model, "./saved_models")
new_model = tf.contrib.saved_model.load_keras_model(saved_model_path)
# The optimizer was not restored, re-attach a new one.
new_model.compile(optimizer='adam',
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])
loss, acc = new_model.evaluate(test_images, test_labels)
print("New model, accuracy: {:5.2f}%".format(100*acc))
