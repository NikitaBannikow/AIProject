# Кросс-платформенная событийно-ориентированная графическая библиотека.
from tkinter import *
import tkinter as tk
import win32gui
import ctypes
# Библиотека алгоритмов компьютерного зрения, обработки изображений и
# численных алгоритмов общего назначения с открытым кодом.
import cv2
from PIL import ImageGrab
# Библиотека для работы с массивами
import numpy as np
# Библиотека обеспечивающая взаимодействие с искусственными нейронными сетями
from tensorflow import keras
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
# Библиотека для визуализации данных c двумерной (2D) графикой
import matplotlib.pyplot as plt

# Загрузка моделей
mlp_model_folder = "out/mlp_model"
mlp_model = keras.models.load_model(mlp_model_folder)
cnn_model_folder = "out/cnn_model"
cnn_model = keras.models.load_model(cnn_model_folder)
vgg16_model_folder = "out/vgg16_model"
vgg16_model = keras.models.load_model(vgg16_model_folder)

def predict_digit(img):
    # Отображение области рисования (захват)
    # plt.imshow(img.convert('RGBA'))
    # plt.show()

    # Изменение рзмера изобржений на 28x28
    img = img.resize((28, 28))
    vgg16_image = img.resize((48, 48))
    # Конвертируем RGB в grayscale
    img = img.convert('L')

    # plt.imshow(img.convert('RGBA'))
    # plt.show()

    # Преобразование и нормализация
    img = np.array(img)
    img = img.reshape(1, 28, 28, 1)
    img = cv2.bitwise_not(img)
    img = img / 255

    vgg16_image = np.array(vgg16_image)
    vgg16_image = vgg16_image.reshape(1, 48, 48, 3)
    vgg16_image = cv2.bitwise_not(vgg16_image)
    vgg16_image = vgg16_image / 255

    # plt.imshow(img[0], cmap=plt.cm.binary)
    # plt.show()

    x = img[0]
    x = np.expand_dims(x, axis=0)

    # print(x)

    mlp_prediction = mlp_model.predict(x)
    cnn_prediction = cnn_model.predict(x)

    mlp = np.argmax(mlp_prediction)
    print(f"MLP: {mlp}")

    cnn = np.argmax(cnn_prediction)
    print(f"CNN: {cnn}")

    v = vgg16_image[0]
    v = np.expand_dims(v, axis=0)
    vgg16_prediction = vgg16_model.predict(v)
    vgg16 = np.argmax(vgg16_prediction)
    print(f"VGG16: {vgg16}")

    return mlp, cnn, vgg16


class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.x = self.y = 0

        # Название окна
        self.title('GUI Predict')

        # Область рисования и результатов
        self.canvas = tk.Canvas(self, width=300, height=300, bg="white", cursor="cross")
        self.label = tk.Label(self, text="Please draw a digit\nand click \'Predict\'", font=("Helvetica", 24))
        self.classify_btn = tk.Button(self, text="Predict", command=self.predict)
        self.button_clear = tk.Button(self, text="Clear", command=self.clear_canvas)

        self.canvas.grid(row=0, column=0, pady=2, sticky=W, )
        self.label.grid(row=0, column=1, pady=2, padx=2)
        self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
        self.button_clear.grid(row=1, column=0, pady=2)

        self.canvas.bind("<B1-Motion>", self.draw_lines)

    def clear_canvas(self):
        # Очистка области рисования
        self.canvas.delete("all")

    def predict(self):
        # Получение координат области рисования
        rect = win32gui.GetWindowRect(self.canvas.winfo_id())
        # Оределелить текущий scale экрана и преобразовать координаты
        scale_factor = ctypes.windll.shcore.GetScaleFactorForDevice(0) / 100
        a, b, c, d = rect
        rect = (a * scale_factor, b * scale_factor, c * scale_factor, d * scale_factor)
        # Захват области рисования
        digit_image = ImageGrab.grab(rect)
        # Получение и отрисовка результатов
        mlp_digit, cnn_digit, vgg16_digit = predict_digit(digit_image)
        self.label.configure(text='MLP: ' + str(mlp_digit) + '\n' + 'CNN: ' + str(cnn_digit) +
                                  '\n' + 'VGG16: ' + str(vgg16_digit))

    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r = 8
        self.canvas.create_oval(self.x - r, self.y - r, self.x + r, self.y + r, fill='black')


app = App()
mainloop()
