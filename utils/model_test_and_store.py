import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
# Библиотека для визуализации данных c двумерной (2D) графикой
import matplotlib.pyplot as plt
# Библиотека обеспечивающая взаимодействие с искусственными нейронными сетями
from tensorflow import keras
# Библиотека для работы с массивами
import numpy as np


# Сохранение и тестирование модели
def execute(model, model_name, x_test_set, y_test_set, y_test_cat):
    print("Testing.....")
    model.evaluate(x_test_set, y_test_cat)

    # Store and load model: https://www.tensorflow.org/guide/keras/save_and_serialize
    # Папка выходных данных
    if not os.path.exists('out'):
        os.makedirs('out')
    model_folder = 'out/' + model_name + '_model'
    print(f"Store model to {model_folder}")
    model.save(model_folder)

    print("Testing reconstructed model...")
    reconstructed_model = keras.models.load_model(model_folder)
    reconstructed_model.evaluate(x_test_set, y_test_cat)

    # Распознавание всей тестовой выборки
    pred = model.predict(x_test_set)
    pred = np.argmax(pred, axis=1)

    # Выделение неверных вариантов
    mask = pred == y_test_set
    x_false = x_test_set[~mask]
    fails_num = x_false.shape[0]
    print(f"Fails number {fails_num}")

    # Вывод первых 25 неверных результатов
    if fails_num > 25:
        fails_num = 25
    plt.figure(figsize=(10, 5))
    for i in range(fails_num):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(x_false[i], cmap=plt.cm.binary)
    plt.show()
