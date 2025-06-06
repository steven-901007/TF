import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.system("cls")


import tensorflow as tf
import numpy as np


data = tf.random.normal([2,3,4,3]) #假設有20張圖片,寬高為30*40,有RGB彩色通道(3)
print(data)
print(data.shape)
print(data.ndim) #維度


re_data = tf.reshape(data,[2*3,4,3])
print(re_data)
print(re_data.shape)
# re_data_lazy = tf.reshape(data,[-1,3,3]) #[]中的-1代表電腦自動計算
# # print(re_data_lazy.shape)
# re_data_2_dim = tf.reshape(data,[1,-1]) #-1就是把沒寫出來的數字都乘起來
# print(re_data_2_dim.shape)
