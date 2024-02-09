# Пирамида Паскаля
from argparse import *

my_parser = ArgumentParser("Введите высоту будущей пирамиды Паскаля: ")
my_parser.add_argument("-high", type=int)  # ввод длины

args = my_parser.parse_args()  #
n = args.high
# n = int(input())

for i in range(1, n + 1):
    for j in range(0, n - i + 1):
        print(' ', end='')
    X = 1
    for j in range(1, i + 1):
        print(' ', X, sep='', end='')
        X = X * (i - j) // j
    print()