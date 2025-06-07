import tensorflow as tf

data = tf.random.uniform([3,10],maxval=10,dtype=tf.int32)
print(data)

top3 = tf.math.top_k(data,k= 4,sorted = True) #回傳前3大值and資料位置
print(top3)