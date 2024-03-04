#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sklearn

from abc import ABC
from abc import abstractmethod
from typing import List, Type

from sklearn.metrics import r2_score


# # Определение основных функций подготовки данных и визуализации для создания моделей Виртуальных Анализаторов

# Ниже описан класс, который следует использовать в качестве шаблона для построения собственных моделей виртуальных анализаторов. Необходимо отнаследовать его и переопределить все абстрактные методы. Изменение метода test() не допускается. На вход модель должна получать numpy.ndarray фиксированной формы {N, C} для Х и {N, 1} для Y, в качестве выхода она также должна предоставлять массив Y той же формы {N, 1}.

# In[2]:


class SoftSensor(ABC):    
    @abstractmethod
    def __init__(self, name: str):
        """Метод инициализации экземпляра, выполняется при создании экземпляра, описанные здесь строки можно выполнить через super().__init__()"""
        self.__model = None
        self.__name = name

    @abstractmethod
    def prepocessing(self):
        """Метод предобработки данных, в него может входить изменение типа данных, изменение формы массива, изменение области значений и так далее """
        pass

    @abstractmethod
    def postprocessing(self):
        """Метод, обратный предобратоки, позволяет вывести значения, похожие на исходные данные"""
        pass

    @abstractmethod
    def evaluate_model(self):
        """Метод обработки данных обученной моделью, принимает на вход numpy массивы Х {N, C} и Y{N,1}, где N - число точек, C - число каналов данных Х"""
        pass

    @abstractmethod
    def train(self):
        """Метод обучения модели по тестовой выборке"""
        pass

    @abstractmethod
    def __str__(self):
        """Метод, срабатывающий при выводе функции print() над экземляром класса, полезен при дебагинге, помогает продемонстрировать работу"""
        pass
    
    def get_model(self):
        """Получение модели экземляра класса"""
        return self.__model

    def set_model(self, model):
        """Установка нового значения в модель экземпляра класса"""
        self.__model = model

    def get_name(self):
        """Получение имени экзмпляра класса"""
        return self.__name

    def set_name(self, name):
        """Установка имени экземпляра класса"""
        self.__name = name

    def test(self, x_test: np.ndarray, y_test: np.ndarray, metric):
        """Метод для обработки тестовой выборки, не подлжежит редактированию в потомках. Возвращает ошибки при неверных типах данных или проблемах в вычислении. 
        Проводит вычисления в порядке: предобработка, вычисление модели, постобработка, вычисление метрики. Возвращает вектор предсказанных значений и метрику качества"""
        if len(x_test.shape) != 2:
            raise AttributeError('Wrong data shape X')
        if y_test.shape[1] != 1 and len(y_test.shape) != 2:
            raise AttributeError('Wrong data shape Y') 

        try:
            x_preproc_values = self.prepocessing(x_test)
            # y_preproc_values = self.prepocessing(y_test)
        except BaseException as err:
            print('Prepocessing error', err)
            raise err
            
        try:
            pred_values = self.evaluate_model(x_preproc_values)
        except BaseException as err:
            print('Model evaluation error', err)
            raise err

        try:
            post_values = self.postprocessing(pred_values)
        except BaseException as err:
            print('Postprocessing error', err)
            raise err
            
        try:
            metric_value = metric.evaluate(post_values, y_test)
        except BaseException as err:
            print('Metric evaluation error', err)
            raise err
            
        if type(post_values) != np.ndarray:
            raise TypeError('Wrong data type after postrocessing')
        if post_values.shape[1] != 1 and len(post_values.shape) != 2:
            raise ValueError('Wrong data shape')

        return post_values, metric_value
        


# Ниже описан абстрактный класс с шаблоном для метрик, нет нужды его использовать для своих моделей, так как мы будем общие классы метрики для всех моделей

# In[3]:


class MetricTemplate(ABC):
    
    @abstractmethod
    def __init__(self, name):
        if type(name) is not str:
            raise AttributeError('Wrong name')
        self.__name = name

    @abstractmethod
    def __call__(self):
        pass

    def evaluate(self, y_pred, y_real):
        try:
            value = self.__call__(y_pred, y_real)
        except BaseException as err:
            print('Metric evaluation error')
            raise err
        if type(value) != float and type(value) != np.float64:
            raise ValueError('Wrong datatype from metric')
        if value.size != 1:
            raise ValueError('Wrong data shape from metric')
        return value
        
    def get_name(self):
        return self.__name


# Класс визуализатора, не требует наследования или переопределения, достаточно создать экземпляр с собственной тестовой выборкой и применить метод visualize() к моделям. Имеет смысл проверить работу своей модели отдельно с его помощью

# In[4]:


class Visualizer():

    def __init__(self, x, y, timestamps, metrics: List[Type[MetricTemplate]], name):
    
        try:
            self.x_test = np.array(x, dtype=np.float64)
            self.y_real = np.array(y, dtype=np.float64)
        except BaseException as err:
            print('Data transform error')
            raise err
        self.timestamps = timestamps
        self.metrics = metrics
        self.name = name
        

    def visualize(self, models: List[Type[SoftSensor]], verbose=True, all=True, each=True):
        metric_num = len(self.metrics)
        metric_dict = {}
        if verbose:
            for metric in self.metrics:
                metric_values = []
                for model in models:
                    _, metric_value = model.test(self.x_test, self.y_real, metric)
                    metric_values.append(metric_value)
                metric_dict[metric.get_name()] = metric_values
            index = []
            for model in models:
                index.append(model.get_name())
            df_metric = pd.DataFrame(data=metric_dict, index=index)
            print(df_metric)

        results = {}
        results['From Lab'] = self.y_real
        for model in models:
            model_pred, _ = model.test(self.x_test, self.y_real, self.metrics[0])
            results[model.get_name()] = np.squeeze(model_pred)

        for name, data in results.items():
            plt.plot(self.timestamps, data, linestyle='', marker=".", label=name)
        plt.title(self.name)
        plt.legend()
        plt.show()
        # res_df = pd.DataFrame(data=results, index=self.timestamps)

        # if all:
        #     plt.plot()
            

        
            
            


# Пример простейшей метрики

# In[5]:


class MSE(MetricTemplate):

    def __init__(self):
        super().__init__('MSE')

    def __call__(self, y_pred, y_real):
        return ((y_pred-y_real)**2).mean().item()


# Класс метрики, используемой для вычисления Коэффициента детерминации, основная используемая нами метрика

# In[6]:


class R2Metric(MetricTemplate):
    def __init__(self):
        super().__init__('Coefficient of determination')

    def __call__(self, y_pred, y_real):
        return sklearn.metrics.r2_score(y_real, y_pred)


# In[ ]:




