import numpy as np
from sklearn.metrics import r2_score

# Функция для вычисления коэффициентов линейной регрессии через SVD и R^2
def linear_regression_svd(X, y):
    # Добавляем столбец из единиц к X для учета свободного члена (intercept)
    X_bias = np.hstack([np.ones((X.shape[0], 1)), X])
    
    # Вычисляем SVD матрицы X_bias
    U, s, Vt = np.linalg.svd(X_bias, full_matrices=False)
    
    # Отбрасываем малые сингулярные значения для стабильности
    tol = max(X_bias.shape) * np.spacing(max(s))
    s_inv = np.array([1/si if si > tol else 0 for si in s])
    
    # Вычисляем псевдообратную матрицу для X_bias
    X_pseudo_inverse = Vt.T @ np.diag(s_inv) @ U.T
    
    # Вычисляем коэффициенты линейной регрессии
    w = X_pseudo_inverse @ y
    
    # Получаем предсказания
    predictions = X_bias @ w
    
    # Вычисляем коэффициент детерминации (R^2)
    r2 = r2_score(y, predictions)
    
    return w, r2

# Тестовые данные
X = np.array([[1, 2], [2, 3], [4, 6], [0, 1], [5, 10]])
y = np.array([1, 2, 3, 0.5, 4.5])

# Обучаем модель линейной регрессии и получаем коэффициенты и R^2
coefficients, r2 = linear_regression_svd(X, y)

# Выводим коэффициенты модели и R^2
print("Коэффициенты линейной регрессии:", coefficients)
print("Коэффициент детерминации (R^2):", r2)