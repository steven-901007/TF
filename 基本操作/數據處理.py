import numpy as np
import pprint as pp
from tensorflow.keras.preprocessing.sequence import pad_sequences


comment1 = [1,2,3,4]
comment2 = [1,2,3,4,5,6,7]
comment3 = [1,2,3,4,5,6,7,8,9,10]

x_train = np.array([comment1,comment2,comment3],dtype=object)
# print(),pp.pprint(x_train)



##補值、縮值
pad = pad_sequences(x_train,value = 99,padding = 'pre', maxlen=5, truncating='pre') 
'''
value = 補數值
padding = 補值位置  'past'(右補)、'pre'(左補)
maxlen = 縮值長度
truncating = 縮值位置  'past'(右補)、'pre'(左補)
'''


print(),pp.pprint(pad)


