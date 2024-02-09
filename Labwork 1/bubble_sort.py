# Пузырьковая сортировка
from argparse import *
# import argparse
from random import *

my_parser = ArgumentParser("Введите длину списка будущего массива: ")
my_parser.add_argument("-l", type=int)  # ввод длины
args = my_parser.parse_args()  #
n = args.l
# n = 5

numbers_list = [random() for i in range(n)]

for i in range(n):
    for j in range(0, n - i - 1):
        if numbers_list[j] > numbers_list[j + 1]:
            numbers_list[j], numbers_list[j + 1] = numbers_list[j + 1], numbers_list[j]
print(numbers_list)