import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
# VGG16
from keras.applications import VGG16
# MNIST dataset
from keras.datasets import mnist
# Библиотека обеспечивающая взаимодействие с искусственными нейронными сетями
from tensorflow import keras
from keras.utils import to_categorical
from keras import models
from keras.layers.core import Dense, Flatten, Dropout
# Библиотека алгоритмов компьютерного зрения, обработки изображений и
# численных алгоритмов общего назначения с открытым кодом.
import cv2
# Библиотека для работы с массивами
import numpy as np
# Вывод информации о модели
import utils.model_info as model_info
# Библиотека для визуализации данных c двумерной (2D) графикой
import matplotlib.pyplot as plt

# Объемная база данных образцов рукописного написания цифр
(x_train, y_train), (x_test, y_test)=mnist.load_data()
# Модель VGG16, веса обучаются ImageNet, размер входных данных модели по умолчанию - 224x224, но минимум - 48x48
# Изменить размер набора данных, преобразовать изображение в оттенках серого в изображение RGB
x_train = [cv2.cvtColor(cv2.resize(i, (48, 48)), cv2.COLOR_GRAY2BGR)for i in x_train]
x_test = [cv2.cvtColor(cv2.resize(i, (48, 48)), cv2.COLOR_GRAY2BGR)for i in x_test]
# Первый шаг: с помощью функции np.newaxis увеличить каждое изображение на одно измерение до (1,48,48,3).
# Второй шаг: соедините каждый массив, чтобы сформировать новый массив x_train через np.concatenate,
# форма массива x_train после соединения будет (10000,48,48,3)
x_train = np.concatenate([arr[np.newaxis]for arr in x_train])
x_test = np.concatenate([arr[np.newaxis]for arr in x_test])


x_train=x_train.astype("float32")/255
x_train=x_train.reshape((60000, 48, 48, 3))
x_test=x_test.astype("float32")/255
x_test=x_test.reshape((10000, 48, 48, 3))
y_train=to_categorical(y_train)
y_test=to_categorical(y_test)

# Тестовый набор
x_val=x_train[:10000]
y_val=y_train[:10000]
x_train=x_train[10000:]
y_train=y_train[10000:]

# Создание модели
conv_base=VGG16(weights='imagenet',
				include_top=False,
				input_shape=(48, 48, 3))
conv_base.trainable=False
model = models.Sequential()
model.add(conv_base)
model.add(Flatten())
model.add(Dense(4096, activation="relu"))
model.add(Dropout(0.5))
# layer 14
model.add(Dense(4096, activation="relu"))
model.add(Dropout(0.5))
# layer 15
model.add(Dense(10, activation="softmax"))
#model.summary()

# Скомпилировать модель
model.compile(
	optimizer="rmsprop",
	loss="categorical_crossentropy",
	metrics=["accuracy"])

# Тренировочная модель
training_history = model.fit(
	x_train,
	y_train,
	batch_size=64,
	epochs=10,
	validation_data=(x_val, y_val))

model_info.dump(model, "vgg16", training_history)

print("Testing.....")
test_loss, test_acc = model.evaluate(x_test, y_test, batch_size=64)
print("The accuracy is:" + str(test_acc))

# Store and load model: https://www.tensorflow.org/guide/keras/save_and_serialize
# Папка выходных данных
if not os.path.exists('out'):
	os.makedirs('out')
model_folder = 'out/vgg16_model'
print(f"Store model to {model_folder}")
model.save(model_folder)

print("Testing reconstructed model...")
reconstructed_model = keras.models.load_model(model_folder)
test_loss, test_acc = reconstructed_model.evaluate(x_test, y_test, batch_size=64)
print("The accuracy is:" + str(test_acc))

# Распознавание всей тестовой выборки
pred = model.predict(x_test)
pred = np.argmax(pred, axis=1)

# Выделение неверных вариантов
mask = pred == x_test
x_false = x_test[~mask]
fails_num = x_false.shape[0]
print(f"Fails number {fails_num}")
