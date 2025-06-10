import numpy as np
from PIL import Image
from glob import glob
import os

## 圖片大小
img_size = (100, 100)
root_path = "C:/Users/steve/python_data/AI/find_nzuko/find_nzuko/"

# 讀取資料路徑 抓出所有 nezuko 和 not-nezuko 的圖片路徑。
nezuko_paths = glob(os.path.join(root_path, "nezuko", "*.PNG"))
not_nezuko_paths = glob(os.path.join(root_path, "not-nezuko", "*.PNG"))

# print(f"Nezuko 數量: {len(nezuko_paths)}")
# print(f"Not Nezuko 數量: {len(not_nezuko_paths)}")

# 預先配置空間
total_images = len(nezuko_paths) + len(not_nezuko_paths)
total_draw_data = np.empty((total_images, *img_size, 3), dtype=np.float32) #儲存所有圖片資料（大小 100x100，RGB三通道）
total_anser_data = np.empty((total_images,), dtype=np.int32) #儲存標籤（0=nezuko, 1=not-nezuko）


## 資料填入
index = 0

for path in nezuko_paths:
    img = Image.open(path).resize(img_size).convert("RGB")
    total_draw_data[index] = np.array(img) / 255.0  # 標準化 0~1
    total_anser_data[index] = 0  # label 為 0（nezuko）
    index += 1

for path in not_nezuko_paths:
    img = Image.open(path).resize(img_size).convert("RGB")
    total_draw_data[index] = np.array(img) / 255.0
    total_anser_data[index] = 1  # label 為 1（not-nezuko）
    index += 1

# print("total_draw:", total_draw_data.shape)
# print("total_anser:", total_anser_data.shape)


from sklearn.model_selection import train_test_split

# 先切割資料（80%訓練 / 20% 測試）
draw_train, draw_test, anser_train, anser_test = train_test_split(
    total_draw_data, total_anser_data, test_size=0.2, random_state=42
) #random_state=42 沒啥特別意義

# print(draw_train.shape)
# print(draw_test.shape)
# print(anser_train.shape)
# print(anser_test.shape)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization


# CNN 建模

model = Sequential()

## Block 1 - 大卷積核提取粗特徵
model.add(Conv2D(64, (7, 7), activation='relu', padding='same', input_shape=(100, 100, 3)))
# model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
## Block 2 - 中等卷積核
model.add(Conv2D(128, (5, 5), activation='relu', padding='same'))
# model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
## Block 3 - 小卷積核堆疊
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
# model.add(BatchNormalization())
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
# model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
## Fully Connected Layers
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.4))  # 降低一點 dropout 強度
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(2, activation='softmax'))  # 二分類輸出
## 編譯模型
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

## 顯示模型結構
model.summary()

from tensorflow.keras.optimizers import Adam

model.compile(
    optimizer=Adam(learning_rate=0.0005),  # 比較深，learning rate 可調小
    loss='sparse_categorical_crossentropy',  # 如果你的 label 是 0/1
    metrics=['accuracy']
)

history = model.fit(
    draw_train, anser_train,
    epochs=30,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

##text

test_loss, test_acc = model.evaluate(draw_test, anser_test)
print(f"測試損失（Loss）：{test_loss:.4f}")
print(f"測試準確率（Accuracy）：{test_acc:.4f}")

predictions = model.predict(draw_test)  # 回傳的是機率值
predicted_classes = (predictions > 0.5).astype("int32")  # 二元分類 → 機率 > 0.5 判為 1


import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = [u'MingLiu'] #細明體
plt.rcParams['axes.unicode_minus'] = False #設定中文
class_names = ["Nezuko", "Not Nezuko"]

# for i in range(10):  # 顯示前 10 張
#     plt.imshow(draw_test[i])
#     plt.title(f"預測：{class_names[predicted_classes[i][0]]} / 正解：{class_names[anser_test[i]]}")
#     plt.axis('off')
#     plt.show()
