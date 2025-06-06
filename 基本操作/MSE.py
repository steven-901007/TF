import tensorflow as tf

row = 5 ##假設的資料量
col = 20


out = tf.random.uniform([row,col])  ##假設機器這樣預測
pre = tf.math.argmax(out,axis = 1) #這只是用來找一組數據的最大值max(list)
print("預測:",out)#假設機器這樣預測
# print(pre)

data = tf.range(row)
data = tf.one_hot(data,depth=col) #建立一個一行只有一個1的list(其他都是0) 
print("資料:",data)#假設正確答案是這個

loss = tf.keras.losses.mse(data,out)
print("loss:",loss)

all_loss = tf.reduce_mean(loss)
print("總體loss:",all_loss)