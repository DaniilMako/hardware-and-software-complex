# Семинар 8 от 24.10.23
# Задачи
# 1. Проведите извлечение признаков из wells_info_with_prod.csv (хоть один из столбцов с датой и категориальным признаком должен остаться). Целевой переменной будет Prod1Year
# 2. Разбейте данные на train и test
# 3. Отмасштабируйте train (в том числе целевую переменную)
# 4. Используя модель масштабирования train отмасштабируйте test (использовать метод transform у той же модели масштабирования)

# import seaborn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import warnings  # для игнорирования ошибок
warnings.filterwarnings('ignore')

# 1. Проведите извлечение признаков из wells_info_with_prod.csv (хоть один из столбцов с датой и категориальным признаком должен остаться). Целевой переменной будет Prod1Year
df = pd.read_csv('wells_info_with_prod.csv')
# print(df.columns)
# 2. Разбейте данные на train и test
columns = ['LATERAL_LENGTH_BLEND', 'LatWGS84', 'Prod1Year']
df = df[columns].copy()
df_train = df.head(40)
df_test = df.tail(20)
# 3. Отмасштабируйте train (в том числе целевую переменную)
scaler = StandardScaler()
df_train[['LATERAL_LENGTH_BLEND', 'LatWGS84', 'Prod1Year']] = scaler.fit_transform(
    df_train[['LATERAL_LENGTH_BLEND', 'LatWGS84', 'Prod1Year']])
print('df_train\n', df_train, '\n\n')
# 4. Используя модель масштабирования train отмасштабируйте test (использовать метод transform у той же модели масштабирования)
df_test[['LATERAL_LENGTH_BLEND', 'LatWGS84', 'Prod1Year']] = scaler.fit_transform(
    df_test[['LATERAL_LENGTH_BLEND', 'LatWGS84', 'Prod1Year']])
print('df_test\n', df_test)


# import pandas as pd
#
# # Загрузка данных
# data = pd.read_csv('wells_info_with_prod.csv')
#
# # Извлечение признаков
# features = data[['CompletionDate', 'StateName']]
# target = data['Prod1Year']
#
# from sklearn.model_selection import train_test_split
#
# # Разделение данных на train и test
# train_features, test_features, train_target, test_target = train_test_split(features, target, test_size=0.2, random_state=42)
#
# from sklearn.preprocessing import StandardScaler
#
# # Создание объекта для масштабирования
# scaler = StandardScaler()
#
# # Масштабирование train_features и train_target
# scaled_train_features = scaler.fit_transform(train_features)
# scaled_train_target = scaler.fit_transform(train_target.values.reshape(-1, 1))
#
# # Масштабирование test_features и test_target с использованием модели масштабирования train
# scaled_test_features = scaler.transform(test_features)
# scaled_test_target = scaler.transform(test_target.values.reshape(-1, 1))
