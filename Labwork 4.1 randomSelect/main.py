import numpy as np
from argparse import *


def replace_elements(array1, array2, p):
    random_numbers = np.random.random(len(array1))  # генерируем случайные числа от 0 до 1 для каждого элемента массива
    return np.where(random_numbers <= p, array2, array1)


my_parser = ArgumentParser(description='Введите полный путь до текстового файла с последовательностью реальных чисел'
                                       'и синтетических чисел')
my_parser.add_argument('filename_1', help='Введите полный путь до входного файла 1 с реальными числами')
my_parser.add_argument('filename_2', help='Введите полный путь до входного файла 2 с синтетическим числами')
my_parser.add_argument('p', help='Введите вероятность P от 0.0 до 1.0 для синтетических чисел из файла 2')
args = my_parser.parse_args()
prob = args.p
print('file_1:', args.filename_1, '  file_2:', args.filename_2, '  p:', args.p)

with open(args.filename_1, 'r') as f:  # записываем массив реальных чисел
    real_num = f.read()
real_num = np.fromstring(real_num, sep=' ')
with open(args.filename_2, 'r') as f:  # записываем массив синтетических чисел
    synth_num = f.read()
synth_num = np.fromstring(synth_num, sep=' ')
print('real_num:', real_num)
print('synth_num', synth_num)

if len(synth_num) != len(real_num):  # проверка, чтоб длины массивов были равны
    raise ValueError("Массивы должны иметь одинаковую длину")
if len(synth_num) > len(real_num):  # проверка, чтоб длина массива реальных не превышала синтетических
    synth_num = synth_num[:len(real_num)]
    print('updated:', synth_num)

# mask = np.isin(real_num, np.abs(synth_num))
# print('mask:', *mask)
# res = np.where(mask, synth_num, real_num)
# print(*res)
print('\nresult:', replace_elements(real_num, synth_num, float(prob)))
