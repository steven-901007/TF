import tensorflow as tf


a = tf.random.uniform((3,10),minval=0,maxval=10,dtype=tf.int32)
b = tf.random.uniform((3,10),minval=0,maxval=10,dtype=tf.int32)

print('a',a)
print('b',b)

print('a==b?',a==b) #是否相同元素
print(a[a==b]) #輸出相同的元素(跟list很像)
print(tf.where(a==b)) #相同元素的位置
