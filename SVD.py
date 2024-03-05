import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import r2_score

# Загрузка данных из Excel-файла
df = pd.read_excel('C:/Users/misha/Soft-Sensors/dannie.xlsx', sheet_name='Sheet1')

# Проверка содержимого DataFrame
if df.empty:
    raise ValueError("DataFrame is empty. Check the file path and sheet name.")

# Конвертация столбцов 'Date' и 'Time' в datetime, учитывая формат
df['DateTime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))

# Выбор двух столбцов данных для анализа
# Предполагается, что после столбцов 'Date' и 'Time' идут данные, которые вы хотите анализировать
data_columns = df.columns[2:4]  # Измените индексы, если нужно выбрать другие столбцы
data_values = df[data_columns]

# Преобразование данных в массивы NumPy для SVD
data_matrix_1 = data_values[data_columns[0]].to_numpy()
data_matrix_2 = data_values[data_columns[1]].to_numpy()

# Применение SVD к каждому столбцу данных
U1, s1, Vt1 = np.linalg.svd(data_matrix_1.reshape(-1, 1), full_matrices=False)
U2, s2, Vt2 = np.linalg.svd(data_matrix_2.reshape(-1, 1), full_matrices=False)

# Визуализация результатов SVD и данных
for i, data_matrix in enumerate([data_matrix_1, data_matrix_2]):
    # Строим график для каждого из столбцов данных
    plt.figure(figsize=(12, 6))
    plt.scatter(df['DateTime'], data_matrix, marker='o', label='Наблюдаемые данные')

    # Визуализация сингулярных значений
    singular_values = s1 if i == 0 else s2
    plt.plot(df['DateTime'], singular_values[0] * U1[:, 0] if i == 0 else singular_values[0] * U2[:, 0], color='red', label='Линия регрессии')

    plt.xlabel('Дата и время')
    plt.ylabel(f'Значения данных ({data_columns[i]})')
    plt.title(f'Линейная регрессия для {data_columns[i]}')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Вычисление коэффициента детерминации R²
    r_squared = r2_score(data_matrix, singular_values[0] * U1[:, 0] if i == 0 else singular_values[0] * U2[:, 0])
    print(f'R² для {data_columns[i]}: {r_squared:.2f}')