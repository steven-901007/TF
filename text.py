import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import time

plt.rcParams['font.sans-serif'] = [u'MingLiu'] #細明體
plt.rcParams['axes.unicode_minus'] = False #設定中文

## 1. 載入 MNIST 資料集
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# print(x_test.shape)

# plt.ion()  # 開啟互動模式
# for n in range(200):
#     plt.clf()
#     plt.imshow(x_test[n].reshape(28, 28), cmap='gray')
#     plt.title(f"正確答案：{y_test[n]}")
#     plt.axis('off')
#     plt.pause(0.5)
# plt.ioff() 
    
## 2. 預處理：標準化到 [0,1]、展平成向量
x_train = x_train.reshape(-1, 28 * 28).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28 * 28).astype("float32") / 255.0

print(x_test.shape)

## 3. 建立模型：簡單的 MLP（多層感知器）
model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 類別：0~9
])

## 4. 編譯模型
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

percrate_list = []
time_use_list = []

epochs = list(range(1, 30))

for nb in epochs: 
    start_time =time.time()
    ## 5. 訓練模型
    model.fit(x_train, y_train, epochs=nb, batch_size=32, validation_split=0.1)
    end_time = time.time()
    ## 6. 測試模型
    test_loss, test_acc = model.evaluate(x_test, y_test)
    # print(f"測試準確率：{test_acc:.4f}")
    percrate_list.append(test_acc)
    time_use_list.append(end_time - start_time)

# 建立圖表與雙 y 軸
fig, ax1 = plt.subplots(figsize=(10, 5))

# 左軸：準確率
color = 'tab:blue'
ax1.set_xlabel("訓練次數（epoch）")
ax1.set_ylabel("準確率（accuracy）", color=color)
ax1.plot(epochs, percrate_list, marker='o', color=color, label="準確率")
ax1.tick_params(axis='y', labelcolor=color)

# 右軸：耗時
ax2 = ax1.twinx()
color = 'tab:orange'
ax2.set_ylabel("耗時（秒）", color=color)
ax2.plot(epochs, time_use_list, marker='s', color=color, label="耗時")
ax2.tick_params(axis='y', labelcolor=color)

# 圖表標題與格線
# fig.suptitle("手抄數字")
fig.tight_layout()
plt.grid(True)
plt.show()