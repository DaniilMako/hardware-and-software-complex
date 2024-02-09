class Pupa:  # поэлементно суммирует
    def __init__(self):  # счетчик зарплаты
        self._salary_counter = 0

    def take_salary(self, amount):  # функция взять зп
        self._salary_counter += amount

    def do_work(self, filename1, filename2):  # выполнить работу с матрицами
        with open(filename1, 'r') as f:  # записываем матрицу 1 в список списков
            matrix1 = [[int(num) for num in line.split()] for line in f]
        with open(filename2, 'r') as f:  # записываем матрицу 2 в список списков
            matrix2 = [[int(num) for num in line.split()] for line in f]
        result = []
        for i in range(len(matrix1)):
            result.append([0] * len(matrix1))
            for j in range(len(matrix1[0])):
                result[i][j] = matrix1[i][j] + matrix2[i][j]
        print('Sum of Pupa:', *result, sep='\n')


class Lupa:  # поэлементно вычитает
    def __init__(self):
        self._salary_counter = 0

    def take_salary(self, amount):
        self._salary_counter += amount

    def do_work(self, filename1, filename2):
        with open(filename1, 'r') as f:  # записываем матрицу 1 в список списков
            matrix1 = [[int(num) for num in line.split()] for line in f]
        with open(filename2, 'r') as f:  # записываем матрицу 2 в список списков
            matrix2 = [[int(num) for num in line.split()] for line in f]
        result = []
        for i in range(len(matrix1)):
            result.append([0] * len(matrix1))
            for j in range(len(matrix1[0])):
                result[i][j] = matrix1[i][j] - matrix2[i][j]
        print('Sub of Lupa:', *result, sep='\n')


class Accountant:
    def give_salary(self, worker):
        worker.take_salary(1000)


pupa = Pupa()
lupa = Lupa()

accountant = Accountant()
accountant.give_salary(pupa)
accountant.give_salary(lupa)

pupa.do_work("matrix1.txt", "matrix2.txt")
lupa.do_work("matrix1.txt", "matrix2.txt")
