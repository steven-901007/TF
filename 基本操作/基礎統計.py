import tensorflow as tf

data = tf.range(12,dtype = tf.float32)
data = tf.reshape(data,[4,3])
# print(data)

#顯示最小最大平均'值'
data_min = tf.reduce_min(data,axis=1)
# print(data_min)
data_max = tf.reduce_max(data,axis=0)
# print(data_max)
data_mean = tf.reduce_mean(data)
print(data_mean)




data = tf.random.uniform([3,10],maxval=1000,minval=0,dtype=tf.int32)
print(data)
#顯示最大最小值的'位置'

arg_max = tf.argmax(data,axis=1)
print(arg_max)
arg_min = tf.argmin(data,axis=0)
print(arg_min)