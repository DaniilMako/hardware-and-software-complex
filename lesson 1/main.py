# 5.9.23
# p.zhikharev@nsu.ru
# П    ДР   КР   З
# 1    2    4    40
# 16   32   12
import os
from random import *
from math import *
from os import *


# Сгенерировать случайное трехзначное число. Вычислить сумму его цифр.
r = randint(100, 999)
print(r)
print(r % 10 + r // 100 + r // 10 % 10)

# Сгенерировать случайное число. Вычислить сумму его цифр.
r = random()
print('random number', r)
# print(r * 10 // 1, r * 10 % 1)
sum = 0
while r != 0:
    sum += r * 10 // 1
    r = r * 10 % 1
print('sum of numbers', sum)

# Задаётся радиус сферы, найти площадь её поверхности и объём.
r = int(input('Введите радиус: '))
print('Площадь:', 4 * pi * r ** 2)
print('Объем:', 4/3 * pi * r ** 3)

# Задаётся год. Определить, является ли он високосным.
year = int(input('Введите год: '))
if year % 400 == 0 and year % 100 != 0 and year % 4 == 0:
    print('ДА')
else:
    print('НЕТ')

# Определить все числа из диапазона 1, N, являющиеся простыми.
for i in range(1, int(input('Введите границу: '))+1):
    cnt = 0
    for j in range(1, i+1):
        if i % j == 0:
            cnt += 1
    if cnt == 2:
        print(i)

# Пользователь делает вклад в размере X рублей сроком на Y лет под 10% годовых (каждый год размер его вклада увеличивается на 10%. Эти деньги прибавляются к сумме вклада, и на них в следующем году тоже будут проценты). Вычислить сумму, которая будет на счету пользователя.
X = int(input('Введите размер вклада(руб): '))
for i in range(1, int(input('Введите срок(лет): '))+1):
    X += X * 0.1
print(X)

# Вывести все файлы, находящиеся в папке и её подпапках с помощью их абсолютных имён. Имя папки задаётся абсолютным или относительным именем. (можно использовать os.walk())
for files in walk('.'):
    print(*files[2], sep='; ')