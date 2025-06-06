import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.system("cls")


import tensorflow as tf
import numpy as np

data = tf.reshape(tf.range(10), (2,5)) #tf.range(10) ==> 拆成兩個list
# print(tf.range(10))
print(data)
print(data.shape)
print(data.ndim)

up_dim = tf.expand_dims(data,axis = 2) #axis =把資料放在ndim裡面的哪個位置
# print(up_dim)
# print(up_dim.shape)
# print(up_dim.ndim)

#只能清除不必要得維度(ndim == 1)
down_dim = tf.squeeze(up_dim,axis = 2)
# print(down_dim)
# print(down_dim.shape)
# print(down_dim.ndim)