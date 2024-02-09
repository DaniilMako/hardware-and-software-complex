from argparse import *


def convolutions(A, B):  # свертка
    result = list()
    for i in range(len(A) - len(B) + 1):  # строки большой матрицы
        result.append([0] * (len(A[0]) - len(B[0]) + 1))  # добавляем новую строку
        for j in range(len(A[0]) - len(B[0]) + 1):  # столбцы большой матрицы
            for u in range(len(B)):  # строки маленькой матрицы
                for v in range(len(B[0])):  # столбцы маленькой матрицы
                    result[i][j] += A[i + u][j + v] * B[u][v]  # Ci,j=∑mx−1u=0∑my−1v=0Ai+u,j+vBu,v
    return result


def output(matrix):  # вывод матрицы
    for i in range(len(matrix)):
        print(matrix[i])


def create_matrix(matrix):  # создание матрицы
    new_matrix = []
    while len(matrix) != 0:
        if matrix[0] == []:
            matrix.pop(0)
            break
        new_matrix.append(matrix.pop(0))
    return new_matrix


my_parser = ArgumentParser(description="Введите путь до текстового файла с двумя матрицами для операции "
                                       "свертки и путь, куда надо сохранить результат работы")
my_parser.add_argument("filename", help="Введите полный путь до входного файла")
my_parser.add_argument("outputFile", help="Введите путь до папки для выходного файла")
args = my_parser.parse_args()
result_file_path = args.outputFile  # путь, по которому надо сохранить файл

with open(args.filename, 'r') as f:  # записываем две матрицы в список
    matrix = [[int(num) for num in line.split()] for line in f]

A = create_matrix(matrix)  # разбиваем исходный список на большую матрицу A
B = create_matrix(matrix)  # и дописываем оставшее в ядро B
output(A), output(B), output(convolutions(A, B))  # вывод на экран матриц A, B и результатирующей

res = convolutions(A, B)
string = ''
for i in range(len(res)):  # переписываем в строку, для записи в txt файл
    string += str(res[i]) + '\n'

with open(result_file_path + "\\outputMatrix.txt", 'w') as result_file:  # записываем в файл по введенному пути
    result_file.write(string)
print("Файл сохранен в:", result_file_path + "\\outputMatrix.txt")
