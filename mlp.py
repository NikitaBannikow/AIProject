import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
# MNIST dataset
import utils.mnist_data as  mnist_data
# Библиотека обеспечивающая взаимодействие с искусственными нейронными сетями
from tensorflow import keras
from tensorflow.python.keras.layers import Dense, Flatten
# Вывод информации о модели
import utils.model_info as model_info
# Сохранение и тестирование модели
import utils.model_test_and_store as model_test_and_store

# Загрузка MIST датасета нормализованных изображений
(x_training_set, y_training_set), (x_test_set, y_test_set) = mnist_data.get_normalized()

# Модель будет иметь 10 выходов, и каждый выход будет соответствовать определенной цифре: от 0 до 9.
y_training_cat = keras.utils.to_categorical(y_training_set, 10)
y_test_cat = keras.utils.to_categorical(y_test_set, 10)

# Создание модели
model = keras.models.Sequential()
model.add(Flatten(input_shape=(28, 28, 1)))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam',
             loss='categorical_crossentropy',
             metrics=['accuracy'])

training_history = model.fit(x_training_set,
          y_training_cat,
          batch_size=32,
          epochs=10,
          validation_split=0.2)

model_info.dump(model, "mlp", training_history)

model_test_and_store.execute(model, "mlp", x_test_set, y_test_set, y_test_cat)
