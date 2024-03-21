import sys
sys.path.append("..")
import Essentials
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

a = np.load('Data_Average.npz', allow_pickle=True)
x1 = a['X_test_2']
x2 = a['X_train_2']

y1 = a['Y_test_2']
y2 = a['Y_train_2']
timestamp1 = y1[:, 1]
timestamp2 = y2[:, 1]

y1 = y1[:, 0].reshape(len(y1), 1)
y1 = y1.astype(np.float64)
y2 = y2[:, 0].reshape(len(y2), 1)
y2 = y2.astype(np.float64)


class TestSoftSensor(Essentials.SoftSensor):
    def __init__(self, x_train, y_train):
        super().__init__('Test')
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.deleted_rows = None
        self.data_size = x_train.shape[1]
        self.fit_scaler(x_train, y_train)
        self.train(x_train, y_train)

    def prepocessing(self, x):
        try:
            trunc_x = np.delete(x, self.deleted_rows, 1)
            return self.x_scaler.transform(trunc_x)
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

    def fit_scaler(self, x_train, y_train):
        self.x_scaler.fit(x_train)
        var = self.x_scaler.var_
        tmp = self.x_scaler.var_ < 0.01
        self.deleted_rows = [i for i, x in enumerate(tmp) if x]
        trunc_x = np.delete(x_train, self.deleted_rows, 1)
        self.x_scaler.fit(trunc_x)
        self.data_size = trunc_x.shape[1]
        self.y_scaler.fit(y_train)

    def evaluate_model(self, x):
        predictions = self.get_model().predict(x)
        return predictions

    def train(self, x_train, y_train):
        model = LinearRegression()
        preproc_y = self.prepocessing(y_train)
        preproc_x = self.prepocessing(x_train)
        model.fit(preproc_x, preproc_y)
        self.set_model(model)

    def __str__(self):
        pass

Test_sensor_1 = TestSoftSensor(x2, y2)
metric = Essentials.R2Metric()
print(Test_sensor_1.test(x1, y1, metric))
test_visual = Essentials.Visualizer(x1, y1, timestamp1, [metric], 'Test SoftSensor R2 metric')