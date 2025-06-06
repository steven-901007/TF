import tensorflow as tf
import numpy as np

data = tf.constant([1,2,3])
print(data)
data = data+1
print(data)

data = tf.broadcast_to(data,(3,3))
print(data)
data = data*10
print(data)