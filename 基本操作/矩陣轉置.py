import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.system("cls")


import tensorflow as tf
import numpy as np


data = tf.range([12])
data = tf.reshape(data,[2,6])

tran_data = tf.transpose(data)
# print(tran_data)

data_43 = tf.range([24])
data_43 = tf.reshape(data_43,[4,3,2])
print(data_43)
tran_data43 = tf.transpose(data_43,perm=[2,1,0])
print(tran_data43)
