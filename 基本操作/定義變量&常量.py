import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.system("cls")

import tensorflow as tf
print(tf.__version__)

print("Eager execution:", tf.executing_eagerly())
a = tf.Variable(2) #int
b = tf.Variable(1.) #float
c = tf.Variable([1.0,2.0,3.0]) #list
d = tf.Variable(1.,dtype=tf.float64) #float

# print(a)
# print(b)
# print(c)
# print(d)


# print(a+b) ##error ==> 不同type不能相加

##結果會從variable變tensor
print(b+c) ##float跟list的每一個值都相加
print(b+c[2]) #取list裡面的值跟b相加

# Tensor 更方便?
aa = tf.constant(3)
bb = tf.constant(2.)
cc = tf.constant([1.0,2.0,3.0])
dd = tf.constant(2.,dtype=tf.float32)