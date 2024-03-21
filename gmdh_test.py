# Подключение файла Essentials, содержащего все необходимые шаблоны

import sys
sys.path.append("..")
import Essentials

# Подключение необходимых для алгоритма библиотек

import numpy as np
from sklearn.preprocessing import StandardScaler
import gmdh
from gmdh import Mia, split_data

# Загрузка данных. Возьмем для примера архив Data_Average

a=np.load('Data_Average.npz', allow_pickle=True)

# Загрузка и подготовка данных

x1=a['X_test_3']
x2=a['X_train_3']

y1=a['Y_test_3']
y2=a['Y_train_3']

timestamp1=y1[:, 1]
timestamp2=y2[:, 1]

y1 = y1[:, 0].reshape(len(y1), 1)
y1 = y1.astype(np.float64)
y2 = y2[:, 0].reshape(len(y2), 1)
y2 = y2.astype(np.float64)

# Покажем, что данные действительно нужной размерности

print(y1.shape, "\n")
print(x1.shape, "\n")

# Создание собственного класса с Виртуальным анализатором на основе шаблона SoftSensor из файла Essentials.py

class gmdh_Mia(Essentials.SoftSensor):
    def __init__(self, x_train, y_train):
        super().__init__('Test')
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.x_scaler.fit(x_train)
        self.y_scaler.fit(y_train)
        self.train(x_train, y_train)

    def prepocessing(self, x):
        try:
            return self.x_scaler.transform(x)
        except:
            try:
                return self.y_scaler.transform(x)
            except BaseException as err:
                print("Ошибка скейлера")
                raise err

    def postprocessing(self, x):
        try:
            return self.x_scaler.inverse_transform(x)
        except:
            try:
                return self.y_scaler.inverse_transform(x)
            except BaseException as err:
                print("Ошибка скейлера")
                raise err

    def evaluate_model(self, x):
        predictions = self.get_model().predict(x)
        return predictions.reshape(-1, 1)

    def train(self, x_train, y_train):
        preproc_y = self.prepocessing(y_train)
        preproc_x = self.prepocessing(x_train)
        model = gmdh.Mia()
        model.fit(preproc_x, preproc_y, k_best=10, test_size=0.26)
        self.set_model(model)

    def __str__(self):
        model=self.get_model()
        return f"Наилучшая найденная модель: \n = {model.get_best_polynomial()}"

# Создание экземпляра класса с алгоритмом gmdh mia

Test_sensor_1=gmdh_Mia(x2, y2)
#print(x2)
# Пример работы метода str

print(Test_sensor_1)

# Создание экземпляра метрики

metric = Essentials.R2Metric()
Test_sensor_1.test(x1, y1, metric)

# Визуализация работы алгоритма

test_visual=Essentials.Visualizer(x1, y1, timestamp1, [metric], 'Test gmdh Mia Sensor R2 metric')
test_visual.visualize([Test_sensor_1])
