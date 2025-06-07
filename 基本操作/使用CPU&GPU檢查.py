import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.system("cls")

import tensorflow as tf
a = tf.Variable(1)
b = tf.Variable(10.)

print(a.device,a) #CUP
print(b.device,b) #GPU

c = tf.constant(10)
d = tf.constant(100.)
# print(c.device,c)
# print(d.device,d)


#只要運算都會在gpu運行
e = a+c
# print(e.device)

#CPU ==> GPU
a_gpu = tf.identity(a)
print(a_gpu.device)