# Отображение первых 25 изображений из MNIST обучающей выборки
# Библиотека для визуализации данных c двумерной (2D) графикой
import matplotlib.pyplot as plt
# MNIST dataset
import mnist_data


(x_training_set, y_training_set), (x_test_set, y_test_set) = mnist_data.get_normalized()

plt.figure(figsize=(10, 5))

for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_training_set[i], cmap=plt.cm.binary)

plt.show()
