import numpy as np
from argparse import *


def replace_elements(r_data, s_data, p):  # заменяем элементы в r_num на s_num с вероятностью p
    random_numbers = np.random.random(len(r_data))  # генерируем случайные числа от 0 до 1 для каждого элемента массива
    return np.where(random_numbers <= p, s_data, r_data)


my_parser = ArgumentParser(description='Введите полный путь до текстового файла с реальными данными'
                                       'и синтетическими данными')
my_parser.add_argument('filename_1', help='Введите полный путь до входного файла 1 с реальными данными')
my_parser.add_argument('filename_2', help='Введите полный путь до входного файла 2 с синтетическим данными')
my_parser.add_argument('p', help='Введите вероятность P от 0.0 до 1.0 для синтетических данны[ из файла 2')
args = my_parser.parse_args()
prob = args.p

with open(args.filename_1, 'r') as f:  # записываем массив реальных данных
    real_data = f.read()
real_data = np.fromstring(real_data, sep=' ')
with open(args.filename_2, 'r') as f:  # записываем массив синтетических данных
    synth_data = f.read()
synth_data = np.fromstring(synth_data, sep=' ')
print('real_num:', real_data)
print('synth_num', synth_data)

if len(synth_data) != len(real_data):  # проверка, чтоб длины массивов были равны
    raise ValueError("Массивы должны иметь одинаковую длину")
if len(synth_data) > len(real_data):  # проверка, чтоб длина массива реальных не превышала длину синтетических
    synth_data = synth_data[:len(real_data)]
    print('изменено:', synth_data)

print('\nresult:', replace_elements(real_data, synth_data, float(prob)))
