# Подключение файла Essentials, содержащего все необходимые шаблоны

import sys
sys.path.append("..")
import Essentials
import Visualizer_pred

# Подключение необходимых для алгоритма библиотек

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
import gmdh
from gmdh import Ria, split_data
from gmdh import CriterionType, SequentialCriterion, Solver, PolynomialType

# Загрузка данных. Возьмем архив Data_Average

a=np.load('../Data_Average.npz', allow_pickle=True)

# Загрузка и подготовка данных

x1=a['X_test_2']
x2=a['X_train_2']
x11=a['X_test_1']
x21=a['X_train_1']

y1=a['Y_test_2']
y2=a['Y_train_2']
y11=a['Y_test_1']
y21=a['Y_train_1']

timestamp1=y1[:, 1]
timestamp2=y2[:, 1]

timestamp11=y11[:, 1]
timestamp21=y21[:, 1]

y1 = y1[:, 0].reshape(len(y1), 1)
y1 = y1.astype(np.float64)
y2 = y2[:, 0].reshape(len(y2), 1)
y2 = y2.astype(np.float64)

y11=y11[:, 0].reshape(len(y11), 1)
y11=y11.astype(np.float64)
y21=y21[:, 0].reshape(len(y21), 1)
y21=y21.astype(np.float64)

# Покажем, что данные действительно нужной размерности

print(y2.shape, "\n")
print(x2.shape, "\n")

print(y21.shape, "\n")
print(y11.shape, "\n")
print(x21.shape, "\n")
print(x11.shape, "\n")

# Создание собственного класса с Виртуальным анализатором на основе шаблона SoftSensor из файла Essentials.py

class gmdh_Ria(Essentials.SoftSensor):
    def __init__(self, x_train, y_train):
        super().__init__('gmdh')
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.x_scaler.fit(x_train)
        self.y_scaler.fit(y_train)
        self.train(x_train, y_train)

    def preprocessing(self, x):
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
        preproc_y = self.preprocessing(y_train)
        preproc_x = self.preprocessing(x_train)
        seq_criterion=SequentialCriterion(criterion_type=CriterionType.SYM_REGULARITY, second_criterion_type=CriterionType.SYM_ABSOLUTE_NOISE_IMMUNITY, solver=Solver.ACCURATE, top=2)
        model = gmdh.Ria()

        # вывод основной информации о параметрах методов и использовании критериев

        #help(seq_criterion)
        #help(model.fit)

        # Метод подбора параметров обучения. При изменении параметров k_best и test_size возможно получить адекватные показатели коэффициента детерминации для каждого набора данных
        # (реализация не закончена, класс gmdh не имеет функции взятия параметров)
        #param_dist = {
        #    'k_best': range(3, 18, 1),
        #    'test_size': [i/100 for i in range(20, 71)],
        #    'limit': [0, 0.01, 0.02, 0.05, 0.1]
        #}
        #model_estimator = model.fit(preproc_x, preproc_y)
        #random_search = RandomizedSearchCV(estimator=model_estimator, param_distributions=param_dist, n_iter=1000, cv=3,
        #                                   n_jobs=-1, verbose=3, scoring='neg_mean_squared_error')
        #random_search.fit(preproc_x, preproc_y)
        #print(random_search.best_params_)


        model.fit(preproc_x, preproc_y, k_best=10, test_size=0.55, n_jobs=-1, verbose=1, limit=0, p_average=2, criterion=seq_criterion, polynomial_type=PolynomialType.QUADRATIC)
        self.set_model(model)

    def __str__(self):
        model=self.get_model()
        return f"Наилучшая найденная модель: \n {model.get_best_polynomial()}"

# Создание экземпляра класса с алгоритмом gmdh mia

Test_sensor_1=gmdh_Ria(x2, y2)
Test_sensor_2=gmdh_Ria(x2, y2)
#print(x2)
# Пример работы метода str

print(Test_sensor_1)

# Создание экземпляра метрики

metric = Essentials.R2Metric()
metric2=Essentials.R2Metric()
Test_sensor_1.test(x1, y1, metric)
Test_sensor_2.test(x1,y1,metric2)

# Визуализация работы алгоритма

test_visual=Visualizer_pred.Visualizer(x1, y1, timestamp1,[metric, metric2], 'Test gmdh Ria Sensor R2 metric')
test_visual.visualize([Test_sensor_1, Test_sensor_2], lines=True, lines_vertical=True)

#test_visual1=Visualizer_pred.Visualizer(x11, y11, timestamp11,[metric, metric2], 'Test gmdh Ria Sensor R2 metric')
#test_visual1.visualize([Test_sensor_1, Test_sensor_2], lines=True)
