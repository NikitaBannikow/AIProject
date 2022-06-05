# Test models from
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Библиотека для работы с массивами
import numpy as np
# Библиотека для визуализации данных c двумерной (2D) графикой
import matplotlib.pyplot as plt
# Библиотека обеспечивающая взаимодействие с искусственными нейронными сетями
from tensorflow import keras
# MNIST dataset
import mnist_data as mnist_data
from keras.datasets import mnist
# Библиотека алгоритмов компьютерного зрения, обработки изображений и
# численных алгоритмов общего назначения с открытым кодом.
import cv2

# Загрузка MIST датасета нормализованных изображений
(x_training_set, y_training_set), (x_test_set, y_test_set) = mnist_data.get_normalized()
# For VGG16
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_test = [cv2.cvtColor(cv2.resize(i, (48, 48)), cv2.COLOR_GRAY2BGR)for i in x_test]
x_test = np.concatenate([arr[np.newaxis]for arr in x_test])
x_test = x_test.astype("float32")/255
x_test = x_test.reshape((10000, 48, 48, 3))

# Загрузка моделей
mlp_model_folder = "out/mlp_model"
mlp_model = keras.models.load_model(mlp_model_folder)
cnn_model_folder = "out/cnn_model"
cnn_model = keras.models.load_model(cnn_model_folder)
vgg16_model_folder = "out/vgg16_model"
vgg16_model = keras.models.load_model(vgg16_model_folder)

# 2223 - 4
# 1233 - 5
# 1638 - 0
# 1200 - 8
# 1204 - 3
n_rec = int(sys.argv[1])
print(f"Predict test image with index {n_rec}")
plt.imshow(x_test_set[n_rec], cmap=plt.cm.binary)
plt.show()

x = x_test_set[n_rec]
x = np.expand_dims(x, axis=0)

# print(x)

mlp_prediction = mlp_model.predict(x)
cnn_prediction = cnn_model.predict(x)

v = x_test[n_rec]
v = np.expand_dims(v, axis=0)
vgg16_prediction = vgg16_model.predict(v)

mlp = np.argmax(mlp_prediction)
print(f"MLP: {mlp}")

cnn = np.argmax(cnn_prediction)
print(f"CNN: {cnn}")

vgg16 = np.argmax(vgg16_prediction)
print(f"VGG16: {vgg16}")
