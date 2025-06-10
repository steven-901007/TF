import numpy as np  
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Activation, BatchNormalization, Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import glob as gb
from PIL import Image
from keras.utils import np_utils
import os 

def show_train_history(train_history, train, validation):
    title = 'Train History'
    output = ""
    if train == "acc":
        title += " - accuracy"
        output = "acc.png"
    else:
        title += " - loss"
        output = "loss.png"

    fig, ax = plt.subplots()
    ax.plot(train_history.history[train])
    ax.plot(train_history.history[validation])

    fig.suptitle(title)
    ax.set_ylabel('train')
    ax.set_ylabel('Epoch')
    ax.legend(['train', 'validation'], loc = 'center right')
    fig.savefig(output, dpi=300)
    plt.close("all")


if __name__ == "__main__":
    # 圖取圖片 分別分為 pikachu 與 not_pikachu
    pikachu = gb.glob("nezuko_500/*.png")
    not_pikachu = gb.glob("not-nezuko_500/*.png")
    totalImages = len(pikachu) + len(not_pikachu)
    index = 0

    train_images = np.empty(shape=[totalImages, 100,100,3], dtype=float)
    train_labels = np.empty(shape=[totalImages, 1], dtype=np.int8)

    # 將圖片數據存入陣列並reshape程 100x100 的RGB三維矩陣 並分別label為 0 1
    for path in pikachu:
        image = Image.open(path)
        image = image.resize((100,100)).convert("RGB")
        image_matrix = np.array(image)
        image_matrix_new = np.reshape(image_matrix, (100, 100, 3))
        train_images[index,:,:,:] = image_matrix_new
        classindex = 0 # 0 is pikachu
        train_labels[index]=classindex
        index += 1

    for path in not_pikachu:
        image = Image.open(path)
        image = image.resize((100,100)).convert("RGB")
        image_matrix = np.array(image)
        image_matrix_new = np.reshape(image_matrix, (100, 100, 3))
        train_images[index,:,:,:] = image_matrix_new
        classindex = 1 # 1 not pikachu
        train_labels[index]=classindex
        index += 1
        

    train_images = train_images / 255.0    # 標準化 0~1
    train_images = train_images.astype('float32')

    # 設定label 並分割20%測試
    train_labels_new = np_utils.to_categorical(train_labels, 2)
    x_Train, x_Test, y_Train, y_Test = train_test_split(train_images, train_labels_new, test_size=0.2)
    x_Train = x_Train.astype('float32')
    

    # CNN 建模
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(7, 7), activation='relu', input_shape=(100,100,3)))
    model.add(Conv2D(256, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Conv2D(288, (3, 3), activation='relu', padding="same"))
    model.add(Conv2D(272, (3, 3), activation='relu', padding="same"))
    model.add(Conv2D(256, (3, 3), activation='relu', padding="same"))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Flatten())
    model.add(Dense(200, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.summary()

    # 使用Adam作為optimizer
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    train_history = model.fit(x_Train, y_Train[:,0], epochs=20, batch_size=64, validation_split=0.2)

    # 畫訓練時間序列圖
    show_train_history(train_history, 'acc', 'val_acc')
    show_train_history(train_history, 'loss', 'val_loss')

    # 利用測試資料計算準確率
    score = model.evaluate(x_Test, y_Test[:,0], verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


    yhat = model.predict_classes(x_Test)

    fig=plt.figure(figsize=(9, 13))
    ax = []
    class_names = ["Not Nezuko", "Nezuko"]
    columns = 5
    rows = 4

    i = 1
    for j in range(20):
        ax.append(fig.add_subplot(rows, columns, i))
        ax[-1].set_title("\n" + class_names[yhat[j]])
        plt.imshow(x_Test[j].astype('float'))
        plt.axis('off')
        i += 1

    plt.show()


