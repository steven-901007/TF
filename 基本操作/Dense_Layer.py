import tensorflow as tf

rows = 10  #資料
sp = 3 #特徵(EX:風速、濕度、溫度)
b = 5 #偏差
net = tf.keras.layers.Dense(b) #建立一個 全連接層（Dense Layer）
net.build((rows,sp))
print('net_w:',net.kernel)
print('net_b:',net.bias)