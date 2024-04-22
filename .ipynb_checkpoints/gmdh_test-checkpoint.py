# Подключение файла Essentials, содержащего все необходимые шаблоны

import sys
sys.path.append("..")
import Essentials
import Visualizer_pred

# Подключение необходимых для алгоритма библиотек

import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
import gmdh
from gmdh import Ria, split_data
from gmdh import CriterionType, SequentialCriterion, Solver, PolynomialType
pd.set_option('display.max_columns', None)  # Не ограничивать количество отображаемых столбцов
pd.set_option('display.max_rows', None)  # Не ограничивать количество отображаемых строк

# Загрузка данных. Возьмем архив Data_Average

data_archive=np.load('../Data_Exponential_Average.npz', allow_pickle=True)

# Загрузка и подготовка данных

def prepare_Y(y):
    timestamp = y[:, 1]
    y = y[:, 0].reshape(len(y), 1)
    y = y.astype(np.float64)

    return y, timestamp

ALL_column_names_1 = data_archive['column_names_1']
all_X_1 = data_archive['all_X_1']
all_Y_1 = data_archive['all_Y_1']

ALL_column_names_2 = data_archive['column_names_2_cat']
all_X_2 = data_archive['all_X_2']
all_Y_2 = data_archive['all_Y_2']

ALL_column_names_3 = data_archive['column_names_3_cat']
all_X_3 = data_archive['all_X_3']
all_Y_3 = data_archive['all_Y_3']

column_names_2 = data_archive['column_names_2']
x_summer_half_2 = data_archive['x_summer_half_2']
y_summer_half_2 = data_archive['y_summer_half_2']
x_winter_half_2 = data_archive['x_winter_half_2']
y_winter_half_2 = data_archive['y_winter_half_2']

column_names_3 = data_archive['column_names_3']
x_summer_half_3 = data_archive['x_summer_half_3']
y_summer_half_3 = data_archive['y_summer_half_3']
x_winter_half_3 = data_archive['x_winter_half_3']
y_winter_half_3 = data_archive['y_winter_half_3']

df_X1 = pd.read_csv(r'../raw_X1.csv', index_col=0)
df_X2 = pd.read_csv(r'../raw_X2.csv', index_col=0)
df_X3 = pd.read_csv(r'../raw_X3.csv', index_col=0)
df_Y1 = pd.read_csv(r'../raw_Y1.csv', index_col=0)
df_Y2 = pd.read_csv(r'../raw_Y1.csv', index_col=0)
df_Y3 = pd.read_csv(r'../raw_Y1.csv', index_col=0)

x_train, x_test, y_train, y_test=train_test_split(all_X_1, all_Y_1, test_size=0.25, random_state=42)

y_train, timestamp_train = prepare_Y(y_train)
sorted_indices_train = np.argsort(timestamp_train)
y_train = y_train[sorted_indices_train]
timestamp_train = timestamp_train[sorted_indices_train]
x_train = x_train[sorted_indices_train]

y_test, timestamp_test = prepare_Y(y_test)
sorted_indices_test = np.argsort(timestamp_test)
y_test = y_test[sorted_indices_test]
timestamp_test = timestamp_test[sorted_indices_test]
x_test = x_test[sorted_indices_test]

print(all_X_1.shape)
print(all_Y_1.shape)
print(x_train.shape)
print(y_test.shape)

x_train_winter, x_test_winter, y_train_winter, y_test_winter = train_test_split(x_winter_half_3, y_winter_half_3, test_size=0.2, random_state=42)
x_train_summer, x_test_summer, y_train_summer, y_test_summer = train_test_split(x_summer_half_3, y_summer_half_3, test_size=0.2, random_state=42)

y_train_summer, timestamp_train_summer = prepare_Y(y_train_summer)
sorted_indices_train_summer = np.argsort(timestamp_train_summer)
y_train_summer = y_train_summer[sorted_indices_train_summer]
timestamp_train_summer = timestamp_train_summer[sorted_indices_train_summer]
x_train_summer = x_train_summer[sorted_indices_train_summer]

y_train_winter, timestamp_train_winter = prepare_Y(y_train_winter)
sorted_indices_train_winter = np.argsort(timestamp_train_winter)
y_train_winter = y_train_winter[sorted_indices_train_winter]
timestamp_train_winter = timestamp_train_winter[sorted_indices_train_winter]
x_train_winter = x_train_winter[sorted_indices_train_winter]

y_test_summer, timestamp_test_summer = prepare_Y(y_test_summer)
sorted_indices_test_summer = np.argsort(timestamp_test_summer)
y_test_summer = y_test_summer[sorted_indices_test_summer]
timestamp_test_summer = timestamp_test_summer[sorted_indices_test_summer]
x_test_summer = x_test_summer[sorted_indices_test_summer]

y_test_winter, timestamp_test_winter = prepare_Y(y_test_winter)
sorted_indices_test_winter = np.argsort(timestamp_test_winter)
y_test_winter = y_test_winter[sorted_indices_test_winter]
timestamp_test_winter = timestamp_test_winter[sorted_indices_test_winter]
x_test_winter = x_test_winter[sorted_indices_test_winter]

column_names_X1 = df_X1.columns.tolist()
column_names_X2 = df_X2.columns.tolist()
column_names_X3 = df_X3.columns.tolist()
column_name_Y1 = 'Давление насыщенных паров в зимний период'
column_name_Y2 = 'Конец кипения легкого бензина'
column_name_Y3 = 'Содержание олефинов в продукте'

column_names_X1.pop()
column_names_X2.pop()
column_names_X3.pop()

feature_names_1 = column_names_X1
feature_names_2 = column_names_X2
feature_names_3 = column_names_X3

target_name_1 = column_name_Y1
target_name_2 = column_name_Y2
target_name_3 = column_name_Y3

#

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
        seq_criterion=SequentialCriterion(criterion_type=CriterionType.REGULARITY, second_criterion_type=CriterionType.STABILITY, solver=Solver.ACCURATE, top=10)
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


        model.fit(preproc_x, preproc_y, k_best=10, test_size=0.45, n_jobs=-1, verbose=0, limit=0, p_average=2, criterion=seq_criterion, polynomial_type=PolynomialType.QUADRATIC)
        self.set_model(model)

    def equation(self, feature_names, target_name):
        model=self.get_model().get_best_polynomial()
        pattern = r'x(\d+)'
        matches = re.findall(pattern, self.get_model().get_best_polynomial())
        for match in matches:
            index = int(match)
            if index < len(feature_names):
                model = model.replace(f'x{match}', f'[{feature_names[index]}]')
                model=model.replace('y =', f'{target_name} =')
        return model

    def __str__(self):
        model=self.get_model()
        return f"Наилучшая найденная модель: \n {model.get_best_polynomial()}"

# Создание экземпляра класса с алгоритмом gmdh mia
print(x_test)
Test_sensor_1=gmdh_Ria(x_train, y_train)
Test_sensor_2=gmdh_Ria(x_train, y_train)
#print(x2)
# Пример работы метода equation (str)

#print(Test_sensor_1)
print(Test_sensor_1.equation(feature_names_1, target_name_1))
#print(type(Test_sensor_1.evaluate_model(x_test)))
# Создание экземпляра метрики

metric = Essentials.R2Metric()
metric2=Essentials.R2Metric()
Test_sensor_1.test(x_test, y_test, metric)
Test_sensor_2.test(x_test,y_test,metric2)

# Визуализация работы алгоритма

test_visual=Visualizer_pred.Visualizer(x_test, y_test, timestamp_test,[metric], 'Test gmdh Ria Sensor R2 metric')
test_visual.visualize([Test_sensor_1, Test_sensor_2], lines=True, lines_vertical=True)

#test_visual1=Visualizer_pred.Visualizer(x11, y11, timestamp11,[metric, metric2], 'Test gmdh Ria Sensor R2 metric')
#test_visual1.visualize([Test_sensor_1, Test_sensor_2], lines=True)
