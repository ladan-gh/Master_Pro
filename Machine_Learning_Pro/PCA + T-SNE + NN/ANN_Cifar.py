import keras, os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.optimizers import Adam

#===========================================
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
import os


def load_batch(f_path, label_key='labels'):

    with open(f_path, 'rb') as f:
        if sys.version_info < (3,):
            d = pickle.load(f)
        else:
            d = pickle.load(f, encoding='bytes')
            # decode utf8
            d_decoded = {}
            for k, v in d.items():
                d_decoded[k.decode('utf8')] = v
            d = d_decoded
    data = d['data']
    labels = d[label_key]

    data = data.reshape(data.shape[0], 3, 32, 32)
    return data, labels

#---------------------------------------------------------------
def load_data(path, negatives=False):

    num_train_samples = 50000

    x_train_local = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
    y_train_local = np.empty((num_train_samples,), dtype='uint8')

    for i in range(1, 6): # Load data_batch_i

        fpath = os.path.join(path, 'data_batch_' + str(i))
        (x_train_local[(i - 1) * 10000: i * 10000, :, :, :],
         y_train_local[(i - 1) * 10000: i * 10000]) = load_batch(fpath)

    fpath = os.path.join(path, 'test_batch')
    x_test_local, y_test_local = load_batch(fpath)

    y_train_local = np.reshape(y_train_local, (len(y_train_local), 1))
    y_test_local = np.reshape(y_test_local, (len(y_test_local), 1))

    if negatives:
        x_train_local = x_train_local.transpose(0, 2, 3, 1).astype(np.float32)
        x_test_local = x_test_local.transpose(0, 2, 3, 1).astype(np.float32)
    else:
        x_train_local = np.rollaxis(x_train_local, 1, 4)
        x_test_local = np.rollaxis(x_test_local, 1, 4)

    return (x_train_local, y_train_local), (x_test_local, y_test_local)

#----------------------------------------
cifar_10_dir = 'C:/Users/Ladan_Gh/PycharmProjects/pythonProject/Machine_Learning_Pro/PCA + T-SNE + NN'

(x_train, y_train), (x_test, y_test) = load_data(cifar_10_dir)

print("Train data (x_train): ", x_train.shape)
print("Train labels (y_train): ", y_train.shape)
print("Test data (x_test): ", x_test.shape)
print("Test labels (y_test): ", y_test.shape)


num_classes = 10

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#----------------------------------------
model = Sequential()

model.add(Conv2D(64, 3, padding="same", activation="relu", input_shape=(32, 32, 3)))
model.add(Conv2D(64, 3, padding="same", activation="relu"))
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(128, 3, activation='relu', padding='same'))
model.add(Conv2D(128, 3, activation='relu', padding='same'))
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(256, 3, activation='relu', padding='same'))
model.add(Conv2D(256, 3, activation='relu', padding='same'))
model.add(Conv2D(256, 3, activation='relu', padding='same'))
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(512, 3, activation='relu', padding='same'))
model.add(Conv2D(512, 3, activation='relu', padding='same'))
model.add(Conv2D(512, 3, activation='relu', padding='same'))
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(512, 3, activation='relu', padding='same'))
model.add(Conv2D(512, 3, activation='relu', padding='same'))
model.add(Conv2D(512, 3, activation='relu', padding='same'))
model.add(MaxPooling2D(2, 2))

model.add(Flatten())
model.add(Dense(4096, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

#----------------------------------------------
# opt = Adam(lr=0.001)
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(x_train, y_train, batch_size=128, epochs=5, validation_data=(x_test, y_test))

#----------------------------------------------
score = model.evaluate(x_test, y_test, verbose=0)

print("Test loss:", score[0])
print("Test accuracy:", score[1])

y_pred = model.predict(x_test)
y_pred2=np.argmax(y_pred, axis=1)
y_test2=np.argmax(y_test, axis=1)
cm = confusion_matrix(y_test2, y_pred2)
print(cm)