# Семинар 5 от 3.10.23
import numpy as np
import pandas as pd

# 1. Создайте DataFrame с 5 столбцами и 10 строками, заполненный случайными числами от 0 до 1. По каждой строке посчитайте среднее чисел, которые больше 0.3.
print('>>>>>>>>>>>>>>>>>>>>ЗАДАНИЕ 1<<<<<<<<<<<<<<<<<<')
df = pd.DataFrame(np.random.random(size=(10, 5)))
print(df[df > 0.3].loc[:], '\n')
st = df.loc[:]
print(df[df > 0.3].mean(axis=1))

# 2. Посчитайте, сколько целых месяцев длилась добыча на каждой скважине в файле wells_info.csv.
print('\n>>>>>>>>>>>>>>>>>>>>ЗАДАНИЕ 2<<<<<<<<<<<<<<<<<<')
df2 = pd.read_csv('wells_info.csv', index_col=0)

df2['CompletionDate'] = pd.to_datetime(df2['CompletionDate'])
df2['FirstProductionDate'] = pd.to_datetime(df2['FirstProductionDate'])
df2['SpudDate'] = pd.to_datetime(df2['SpudDate'])
print((df2['CompletionDate'].dt.to_period('M') - df2['SpudDate'].dt.to_period('M')))

# 3. Заполните пропущенные числовые значения медианой, а остальные самым часто встречаемым значением в файле wells_info_na.csv
print('\n>>>>>>>>>>>>>>>>>>>>ЗАДАНИЕ 3<<<<<<<<<<<<<<<<<<')
df3 = pd.read_csv('wells_info_na.csv')
print('Таблица ДО:\n', df3)
# заполнение пропущеных числовых значений медианой (по столбцу)
df3['LatWGS84'] = df3['LatWGS84'].fillna(df3['LatWGS84'].median())
df3['LonWGS84'] = df3['LonWGS84'].fillna(df3['LonWGS84'].median())
df3['PROP_PER_FOOT'] = df3['PROP_PER_FOOT'].fillna(df3['PROP_PER_FOOT'].median())
# сначала заполнение нечисловых значений часто встречаемым значением
df3 = df3.fillna(df3.stack().value_counts().idxmax())  # преобразуем, считаем, находим, заполняем

print('Таблица ПОСЛЕ:\n', df3)
