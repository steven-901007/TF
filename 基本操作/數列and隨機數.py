import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.system("cls")

import tensorflow as tf
import numpy as np

##將list轉成tensor
a = np.zeros(12)
a = tf.convert_to_tensor(a)
# print(a)

##建立tensor list
b = tf.ones(12) #一維
# print(b)
c = tf.zeros([5,3]) #二維
# print(c) 
d = tf.ones([5,4,3]) #三維
# print(d)

e = tf.zeros_like(b) #生成一個向b格式的全0資料
# print(e)
f = tf.fill([5,3],7.) #建立一個list並填上數字7
# print(f)

#建立random number
g = tf.random.normal([100]) #隨機生成
# print(g)
h = tf.random.uniform([100],minval = 0,maxval = 10,dtype=tf.int64) #有範圍的隨機數 並指定其中維整數或小數(dtype)
# print(h)

#建立數列
i = tf.range([10],dtype=tf.float32)
# print(i)
j = tf.random.shuffle(i) #將數列打亂
# print(j)

