# Demonstrate eager execution for more interactive output
import tensorflow as tf
import numpy as np

print("Using TensorFlow: ", tf.__version__)

# executes tf operations immediately
tf.enable_eager_execution()

# Basic Operators
print(tf.add(1, 2))
print(tf.add([1, 2], [3, 4]))
print(tf.square(5))
print(tf.reduce_sum([1, 2, 3]))
print(tf.encode_base64("hello world"))

# Operator overloading is also supported
print(tf.square(2) + tf.square(3))

# Auto conversion to and from numpy arrays
ndarray = np.ones([3, 3])
