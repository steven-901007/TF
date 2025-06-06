#2范數 -- 平方合開根號
import tensorflow as tf


data = tf.constant([3,4],dtype=tf.float32)
# print(data)

nm_two = tf.norm(data,ord = 2) #ord默認2
# print(nm_two)


#1范數 -- 所有值的絕對值的和

data = tf.range(12,dtype = tf.float32)
data = tf.reshape(data,[4,3])
data = data-6
print(data)

nm_one = tf.norm(data,ord = 1)
# print(nm_one)

nm_one_ax = tf.norm(data,ord = 1,axis = 0) #axis在某一層求1范數
print(nm_one_ax)