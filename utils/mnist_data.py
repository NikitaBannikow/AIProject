# Вспомогательный модуль для загрузки MNIST dataset
import os
# Объемная база данных образцов рукописного написания цифр
from keras.datasets import mnist

def get_normalized():
    (x_training_set, y_training_set), (x_test_set, y_test_set) = mnist.load_data()
    print("The training MNIST dataset consists:")
    print("-----------------------------------")
    # Каждое изображение имеет размер 28х28 пикселей и представлено в градациях серого,
    # т.е. каждый пиксел имеет значение от 0 до 255 (0 – черный цвет, 255 – белый)
    # Изображения цифр обучающей выборки
    print('x_training_set:', x_training_set.shape)
    # Вектор соответствующих значений цифр обучающей выборки
    print('y_training_set:', y_training_set.shape)
    # Изображения цифр тестовой выборки
    print('x_test_set:', x_test_set.shape)
    # Вектор соответствующих значений цифр для тестовой выборки
    print('y_test_set:', y_test_set.shape)
    print

    # Нормализация входных данных из значений 0..255 в 0..1
    x_training_set = x_training_set / 255
    x_test_set = x_test_set / 255

    return (x_training_set, y_training_set), (x_test_set, y_test_set)
