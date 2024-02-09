# Семинар от 24.10.23
# Задачи
# 1. Проведите извлечение признаков из wells_info_with_prod.csv (хоть один из столбцов с датой и категориальным признаком должен остаться). Целевой переменной будет Prod1Year
# 2. Разбейте данные на train и test
# 3. Отмасштабируйте train (в том числе целевую переменную)
# 4. Используя модель масштабирования train отмасштабируйте test (использовать метод transform у той же модели масштабирования)


import pandas as pd
from sklearn.preprocessing import StandardScaler  # для модели
import warnings  # для игнорирования ошибок
warnings.filterwarnings('ignore')

# 1. Проведите извлечение признаков из wells_info_with_prod.csv (хоть один из столбцов с датой и категориальным признаком должен остаться). Целевой переменной будет Prod1Year
df = pd.read_csv('wells_info_with_prod.csv')
# 2. Разбейте данные на train и test
columns = ['LATERAL_LENGTH_BLEND', 'LatWGS84', 'Prod1Year']
df = df[columns].copy()
df_train = df.head(40)
df_test = df.tail(20)
print(df_train.head(5))
# 3. Отмасштабируйте train (в том числе целевую переменную)
scaler = StandardScaler()
df_train[['LATERAL_LENGTH_BLEND', 'LatWGS84', 'Prod1Year']] = scaler.fit_transform(
    df_train[['LATERAL_LENGTH_BLEND', 'LatWGS84', 'Prod1Year']])
print('df_train\n', df_train.head(5), '\n\n')
# 4. Используя модель масштабирования train отмасштабируйте test (использовать метод transform у той же модели масштабирования)
df_test[['LATERAL_LENGTH_BLEND', 'LatWGS84', 'Prod1Year']] = scaler.fit_transform(
    df_test[['LATERAL_LENGTH_BLEND', 'LatWGS84', 'Prod1Year']])
print('df_test\n', df_test.head(5))
