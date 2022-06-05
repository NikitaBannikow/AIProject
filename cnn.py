import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
# Библиотека для визуализации данных c двумерной (2D) графикой
import numpy as np
# MNIST dataset
import utils.mnist_data as mnist_data
# Библиотека обеспечивающая взаимодействие с искусственными нейронными сетями
from tensorflow import keras
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
# Вывод информации о модели
import utils.model_info as model_info
# Сохранение и тестирование модели
import utils.model_test_and_store as model_test_and_store


# Загрузка MIST датасета нормализованных изображений
(x_training_set, y_training_set), (x_test_set, y_test_set) = mnist_data.get_normalized()

# Модель будет иметь 10 выходов, и каждый выход будет соответствовать определенной цифре: от 0 до 9.
y_training_cat = keras.utils.to_categorical(y_training_set, 10)
y_test_cat = keras.utils.to_categorical(y_test_set, 10)

# Для сверточной нейронной сети множества x_training_set и x_test_set нужно дополнительно подготовить.
# На входе такой сети ожидается четырехмерный тензор в формате:
# - (batch, channels, rows, cols) – если data_format = 'channels_first';
# - (batch, rows, cols, channels) – если data_format = 'channels_last'.
# Где channels – это каналы на входах сверточных слоев
# То есть, входные данные должны иметь размерность: (batch, rows = 28, cols = 28, channels = 1)
x_training_set = np.expand_dims(x_training_set, axis=3)
x_test_set = np.expand_dims(x_test_set, axis=3)

# Создание модели
model = keras.models.Sequential()
model.add(Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2), strides=2))
model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
model.add(MaxPooling2D((2, 2), strides=2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10,  activation='softmax'))

model.compile(optimizer='adam',
             loss='categorical_crossentropy',
             metrics=['accuracy'])

training_history = model.fit(x_training_set,
          y_training_cat,
          batch_size=32,
          epochs=10,
          validation_split=0.2)

model_info.dump(model, "cnn", training_history)

model_test_and_store.execute(model, "cnn", x_test_set, y_test_set, y_test_cat)
