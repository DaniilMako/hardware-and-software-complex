# Лабораторная 5.1
# Данные, которые предоставил кинотеатр находятся в файлах cinema_sessions.csv и titanic_with_labels
# 1. Пол (sex): отфильтровать строки, где пол не указан, преобразовать оставшиеся в число 0/1;
# 2. Номер ряда в зале (row_number): заполнить вместо NAN максимальным значением ряда;
# 3. Количество выпитого в литрах (liters_drunk): отфильтровать отрицательные значения и нереально большие значения (выбросы). Вместо них заполнить средним
import pandas as pd

# считываем таблицу и устанавливаем разделитель - пробел
df = pd.read_csv('titanic_with_labels.csv', delimiter=' ')

# отфильтровываем бесполых
df = df[(df['sex'] != '-') & (df['sex'] != 'Не указан')]


def is_female(df_temp):  # преобразование: если мужчина - 0, женщина - 1
    return 1 if df_temp['sex'] == 'M' or df_temp['sex'] == 'м' else 0


df['sexMF'] = df.apply(is_female, axis=1)

# заполняем пустые макс значением ряда
df['row_number'] = df['row_number'].fillna(df['row_number'].max())

# отфильтровываем отрицательные литры
df = df[df['liters_drunk'] > 0]

# и отфильтровываем нереально большие литры на средние значения
df.loc[df['liters_drunk'] > 5, 'liters_drunk'] = pd.NA  # нереально большие литры = NaN
mean_twl = df['liters_drunk'].mean()  # вычисляем среднее по столбцу без нереально больших литров
print(mean_twl)
df['liters_drunk'] = df['liters_drunk'].fillna(mean_twl)  # заполняем пустые ячейки на среднее значение

print('>>>>>>>titanic_with_labels\n', df)
