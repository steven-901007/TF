import tensorflow as tf

zero_ten = tf.zeros([2,4,3])
one_ten = tf.ones([2,4,3])
print(zero_ten)
print(one_ten)

#dim 沒變
con = tf.concat([zero_ten,one_ten],axis = 0) #axis合併位置shape(0,1,2)
# print(con) #shape = (4,4,3)

#dim+1
st = tf.stack([zero_ten,one_ten],axis = 0)
print(st) #shape(2,2,4,3)

unst = tf.unstack([zero_ten,one_ten],axis = 0)
print(unst) #shape(2,4,3)*2