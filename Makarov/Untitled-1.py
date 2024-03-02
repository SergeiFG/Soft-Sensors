from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import r2_score

class SoftSensor(ABC):    
    @abstractmethod
    def __init__(self, name: str):
        self.__model = None
        self.__name = name

    @abstractmethod
    def preprocessing(self, x):
        pass

    @abstractmethod
    def postprocessing(self, x):
        pass

    @abstractmethod
    def evaluate_model(self, x):
        pass

    @abstractmethod
    def train(self, x_train, y_train):
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


# реализация линейной регрессии
class LinearRegressionSoftSensor(SoftSensor):
    def __init__(self, name: str):
        super().__init__(name)
    
    def preprocessing(self, x):
        # Добавляем столбец из единиц к X для учета свободного члена
        return np.hstack([np.ones((x.shape[0], 1)), x])

    def postprocessing(self, x):
        # В этом примере линейной регрессии постобработка не требуется
        return x

    def evaluate_model(self, x):
        if self.get_model() is None:
            raise ValueError("Model is not trained yet.")
        return x @ self.get_model()

    def train(self, x_train, y_train):
        X_bias = self.preprocessing(x_train)
        U, s, Vt = np.linalg.svd(X_bias, full_matrices=False)
        tol = max(X_bias.shape) * np.spacing(max(s))
        s_inv = [1/si if si > tol else 0 for si in s]
        X_pseudo_inverse = Vt.T @ np.diag(s_inv) @ U.T
        w = X_pseudo_inverse @ y_train
        self.set_model(w)

    def __str__(self):
        return f"LinearRegressionSoftSensor(name={self.get_name()}, model={self.get_model()})"

# Использование нового класса
name = "Linear Soft Sensor"
linear_sensor = LinearRegressionSoftSensor(name)
X = np.array([[1, 2], [2, 3], [4, 6], [0, 1], [5, 10]])
y = np.array([1, 2, 3, 0.5, 4.5]).reshape(-1, 1)
linear_sensor.train(X, y)
print(linear_sensor)