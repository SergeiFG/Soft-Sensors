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

# In[2]:


class SoftSensor(ABC):    
    @abstractmethod
    def __init__(self, name: str):
        self.__model = None
        self.__name = name

    @abstractmethod
    def prepocessing(self):
        pass

    @abstractmethod
    def postprocessing(self):
        pass

    @abstractmethod
    def evaluate_model(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def __str__(self):
        pass
    
    def get_model(self):
        return self.__model

    def set_model(self, model):
        self.__model = model

    def get_name(self):
        return self.__name

    def set_name(self, name):
        self.__name = name

    def test(self, x_test: np.ndarray, y_test: np.ndarray, metric):
        if len(x_test.shape) != 2:
            raise AttributeError('Wrong data shape X')
        if y_test.shape[1] != 1 and len(y_test.shape) != 2:
            raise AttributeError('Wrong data shape Y') 

        try:
            x_preproc_values = self.prepocessing(x_test)
            y_preproc_values = self.prepocessing(y_test)
        except BaseException as err:
            print('Prepocessing error', err)
            raise err
            
        try:
            pred_values = self.evaluate_model(x_preproc_values)
        except BaseException as err:
            print('Model evaluation error', err)
            raise err

        try:
            metric_value = metric.evaluate(pred_values, y_preproc_values)
        except BaseException as err:
            print('Metric evaluation error', err)
            raise err

        try:
            post_values = self.postprocessing(pred_values)
        except BaseException as err:
            print('Postprocessing error', err)
            raise err
            
        if type(post_values) != np.ndarray:
            raise TypeError('Wrong data type after postrocessing')
        if post_values.shape[1] != 1 and len(post_values.shape) != 2:
            raise ValueError('Wrong data shape')

        return post_values, metric_value
        


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
            

        
            
            


# In[ ]:





# In[5]:


class MSE(MetricTemplate):

    def __init__(self):
        super().__init__('MSE')

    def __call__(self, y_pred, y_real):
        return ((y_pred-y_real)**2).mean().item()


# In[6]:


class R2Metric(MetricTemplate):
    def __init__(self):
        super().__init__('Coefficient of determination')

    def __call__(self, y_pred, y_real):
        return sklearn.metrics.r2_score(y_real, y_pred)


# In[ ]:




