import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
# Библиотека для визуализации данных c двумерной (2D) графикой
import matplotlib.pyplot as plt
# Библиотека обеспечивающая взаимодействие с искусственными нейронными сетями
from tensorflow import keras


# Дамп информации о модели
def dump(model, model_name, training_history):
    # Папка выходных данных
    if not os.path.exists('out'):
        os.makedirs('out')

    # (Опционально) Визуализация структуры модели нейронной сети
    # Требует установки пакета pydot и graphviz
    model_file = 'out/' + model_name + '_model.png'
    keras.utils.plot_model(
        model,
        to_file=model_file,
        show_shapes=True,
        show_layer_names=True,
    )
    print(f"Graphical model stored to file {model_file}")
    print("Structure of model:")
    print(model.summary())

    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.title('Loss history')
    plt.plot(training_history.history['loss'], label='training set')
    plt.plot(training_history.history['val_loss'], label='validation set')
    plt.legend()
    plt.show()

    plt.xlabel('Epoch Number')
    plt.ylabel('Accuracy')
    plt.title('Accuracy history')
    plt.plot(training_history.history['accuracy'], label='training set')
    plt.plot(training_history.history['val_accuracy'], label='validation set')
    plt.legend()
    plt.show()
