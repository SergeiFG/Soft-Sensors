import Essentials
from Essentials import SoftSensor, MetricTemplate, MSE
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sklearn
import matplotlib.ticker as ticker
import matplotlib.dates as mdates

from abc import ABC
from abc import abstractmethod
from typing import List, Type
from matplotlib.gridspec import GridSpec
from tabulate import tabulate

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

    def visualize(self, models: List[Type[SoftSensor]], verbose=True, all=True, each=True, lines=True, lines_vertical=False):
        metric_num = len(self.metrics)
        metric_dict = {}
        if verbose:
            for metric in self.metrics:
                metric_values = []
                metric_mse = Essentials.MSE()
                mse=[]
                for model in models:
                    y_pred, _ = model.test(self.x_test, self.y_real, metric)
                    #print(y_pred)
                    _, metric_value = model.test(self.x_test, self.y_real, metric)
                    mse_value=metric_mse(y_pred, self.y_real)
                    mse.append(mse_value)
                    #mse.append(metric1(y_pred, self.y_real))
                    metric_values.append(metric_value)
                #print(metric1.get_name())
                metric_dict[metric.get_name()] = metric_values
                metric_dict[metric_mse.get_name()]=mse
            index = []
            errors=[]
            column_name='Errors'

            for model in models:
                index.append(model.get_name())
            #print(metric_dict)
            df_metric = pd.DataFrame(data=metric_dict, index=index)
            data = [list(df_metric.columns)]
            data += df_metric.values.tolist()
            headers=list(df_metric.columns)
            #for row in data:
            #    value=mse()
            #    errors.append(value)
            for i, model_name in enumerate(index):
                data[i + 1].insert(0, model_name)
            headers.append(column_name)
            #print(data)
            #for i in range(len(data)):
            #    print(data[i])
            #    data[i].append(errors[i])
            #for i, header in enumerate(headers):
                #data[0].insert(i, header)

            table = tabulate(data, headers='firstrow', tablefmt='grid')
            print(table)
            #print(df_metric)
        results = {}
        real=self.y_real
        if real.ndim==2:
            real=np.squeeze(real)
        #print(real.shape)
        #print(real.shape)
        #print(real.shape)
        results['From Lab'] = real
        for model in models:
            model_pred, _ = model.test(self.x_test, self.y_real, self.metrics[0])
            results[model.get_name()] = np.squeeze(model_pred)
        res = {}
        res['From Lab'] = self.y_real
        for model in models:
            model_pred, _ = model.test(self.x_test, self.y_real, self.metrics[0])
            res['Arrange']=np.squeeze(np.subtract(model_pred, res['From Lab']))

        #fig1, ax1 = plt.subplots()
        #fig2, ax2 = plt.subplots()
        #fig3, ax3 = plt.subplots()
        #fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(8, 10))
        fig = plt.figure(figsize=(10, 8))
        grid = GridSpec(3, 1)
        ax1 = fig.add_subplot(grid[0, 0])
        ax2 = fig.add_subplot(grid[1, 0])
        ax3 = fig.add_subplot(grid[2, 0])
        for name, data in results.items():
            #print(data.shape)
            #print(type(data))
            ax1.plot(self.timestamps, data, linestyle='', marker=".", markersize=8, label=name)
            ax2.plot(res['From Lab'], res['Arrange'], linestyle='', marker='.')
            ax3.plot(results['From Lab'], results[model.get_name()], linestyle='', marker='.')
            #ax2.plot()
            #print(data)
        if lines:
            for i in range(len(self.timestamps)-1):
                ax1.plot([self.timestamps[i], self.timestamps[i+1]], [results['From Lab'][i], results['From Lab'][i+1]], color='blue',
                         linestyle='-', linewidth=0.6)
                ax1.plot([self.timestamps[i], self.timestamps[i + 1]],
                         [results[model.get_name()][i], results[model.get_name()][i + 1]], color='orange',
                         linestyle='-', linewidth=0.6)
        if lines_vertical:
            for i in range(len(self.timestamps)):
                ax1.plot([self.timestamps[i], self.timestamps[i]], [results['From Lab'][i], results[model.get_name()][i]], color='gray',
                         linestyle='-')
        x=np.linspace(min(real), max(real), 100)
        y=x
        ax3.plot(x,y, linestyle='-', color='green', linewidth=0.6)
        # Устанавливаем параметры визуализации
        ax1.set_title(self.name)
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Count', rotation=90)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax1.xaxis.set_minor_formatter(mdates.AutoDateFormatter('%Y-%m'))
        ax1.xaxis.set_tick_params(rotation=30)
        ax1.legend(loc='best')

        ax2.set_title("Model prediction errors")
        ax2.set_xlabel('Real parameters')
        ax2.set_ylabel('Error', rotation=90)
        #ax2.set_xticks(rotation=45)

        ax3.set_title('Real-predicted relation')
        ax3.set_xlabel('Real')
        ax3.set_ylabel('Predicted', rotation=90)
        ax3.xaxis.set_tick_params(rotation=0)
        #ax3.legend(loc='best')

        ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax1.xaxis.set_minor_locator(mdates.AutoDateLocator()) #Должен устанавливать неглавные вертикальные оси, но вот не надо это использовать, если слишком большой/неравномерный временной промежуток
        ax1.xaxis.set_minor_formatter(mdates.DateFormatter(''))
        ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        ax1.grid(True, axis='y', which='major', linestyle='-', linewidth=1)
        ax1.grid(True, axis='y', which='minor', linestyle=':', linewidth=0.8)
        ax1.xaxis.grid(True, which='major', linestyle='--', linewidth=1)
        ax1.xaxis.grid(True, which='minor', linestyle=':', linewidth=0.8)
        ax1.yaxis.set_label_coords(-0.1, 0.5)
        ax1.xaxis.set_label_coords(1, -0.2)

        ax2.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        ax2.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        ax2.grid(True, axis='y', which='major', linestyle='-', linewidth=1)
        ax2.grid(True, axis='y', which='minor', linestyle=':', linewidth=0.8)
        ax2.grid(True, axis='x', which='major', linestyle='--', linewidth=1)
        ax2.grid(True, axis='x', which='minor', linestyle=':', linewidth=0.8)
        ax2.yaxis.set_label_coords(-0.1, 0.5)
        ax2.xaxis.set_label_coords(1, -0.2)

        ax3.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        ax3.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        ax3.grid(True, axis='y', which='major', linestyle='-', linewidth=1)
        ax3.grid(True, axis='y', which='minor', linestyle=':', linewidth=0.8)
        ax3.grid(True, axis='x', which='major', linestyle='--', linewidth=1)
        ax3.grid(True, axis='x', which='minor', linestyle=':', linewidth=0.8)
        ax3.yaxis.set_label_coords(-0.1, 0.5)
        ax3.xaxis.set_label_coords(1, -0.2)

        plt.subplots_adjust(hspace=0.8)
        #plt.tight_layout()
        plt.show()
        #print(results['From Lab'])

        # res_df = pd.DataFrame(data=results, index=self.timestamps)

        # if all:
        #     plt.plot()