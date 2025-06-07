import tensorflow as tf


data = tf.random.shuffle(tf.range(10))
# print(data)


#排列
up_inf = tf.sort(data,direction = 'ASCENDING') #生序
# print(up_inf)
down_inf = tf.sort(data,direction = 'DESCENDING') #降序
# print(down_inf)

#排序返回索引位置
up_lc = tf.argsort(data,direction = 'ASCENDING')
# print(up_lc)
down_lc = tf.argsort(data,direction = 'DESCENDING')
# print(down_inf)


data_2dim = tf.random.uniform([3,5],maxval=10,minval=0,dtype=tf.int32)
print(data_2dim)


#排列
up_inf_2dim = tf.sort(data_2dim,axis=1,direction = 'ASCENDING')
print(up_inf_2dim)
down_inf_2dim = tf.sort(data_2dim,axis=0,direction = 'DESCENDING')
print(down_inf_2dim)


#排序返回索引位置
up_lc_2dim = tf.argsort(data_2dim,direction = 'ASCENDING')
print(up_lc_2dim)
down_lc_2dim = tf.argsort(data_2dim,direction = 'DESCENDING')
print(down_inf_2dim)
