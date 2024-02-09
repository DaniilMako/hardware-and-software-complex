# Лабораторная работа 10. Матчасть DL
# Задача: реализовать и обучить нейронную сеть, состоящую из 2 нейронов, предсказывать значения функции XOR.
# При выполнении лабораторной запрещается использовать фреймворки для глубокого обучения (как PyTorch, Tensorflow, Caffe, Theano и им подобные).
import numpy as np


# Прямой проход включает в себя вычисление прогнозируемого выходного сигнала, который является функцией взвешенной суммы входных данных, предоставленных нейронам:
def sigmoid(x):  # сигмоидальная функция - функция активации
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):  # производная от сигмоиды
    return x * (1 - x)


# Ошибка может быть просто записана как разница между прогнозируемым результатом и фактическим результатом. Математически:
def loss(t, y):  # где t - целевой / ожидаемый результат, а y - прогнозируемый результат
    return t - y


# 1. Класс Neuron, имеющий вектор весов self._weigths
class Neuron:
    def __init__(self):
        self._weights = np.random.randn(2)
        self.bias = np.random.randn()

    # 2. Два метода класса Neuron: forward(x), backward(x, loss) - реализующих прямой и обратный проход по нейронной сети.
    # Метод forward должен реализовывать логику работу нейрона: умножение входа на вес self._weigths, сложение и функцию активации сигмоиду.
    def forward(self, x):  # прямой проход
        res_mul = np.dot(x, self._weights) + self.bias
        return sigmoid(res_mul)


    # Метод backward должен реализовывать взятие производной от сигмоиды и используя состояние нейрона обновить его веса.
    def backward(self, x):  # обратный проход
        return sigmoid_derivative(x)


# 3. Реализовать с помощью класса Neuron нейронную сеть с архитектурой из трёх нейронов, предложенную в статье
class Model:
    def __init__(self):
        self.neurons = [Neuron(), Neuron(), Neuron()]

    # 4. Для красоты обернуть в класс Model с методами forward и backward, реализующими правильное взаимодействие нейронов на прямом и обратном проходах.
    def forward(self, x):  # прямой проход
        hidden_layer = np.array([self.neurons[0].forward(np.array(x)), self.neurons[1].forward(np.array(x))])
        hidden_layer_output = self.neurons[2].forward(hidden_layer)
        return hidden_layer_output

    def backward(self, x, error):  # обратный проход
        d_predicted_output = error * self.neurons[2].backward(self.forward(x))

        error_hidden_layer0 = d_predicted_output * self.neurons[2]._weights[0]
        d_hidden_layer0 = error_hidden_layer0 * self.neurons[0].backward(self.neurons[0].forward(x))

        error_hidden_layer1 = d_predicted_output * self.neurons[2]._weights[1]
        d_hidden_layer1 = error_hidden_layer1 * self.neurons[1].backward(self.neurons[1].forward(x))
        # обновление весов и смещений
        self.neurons[0]._weights += np.array(x) * d_hidden_layer0
        self.neurons[0].bias += np.sum(d_hidden_layer0)

        self.neurons[1]._weights += np.array(x) * d_hidden_layer1
        self.neurons[1].bias += np.sum(d_hidden_layer1)

        self.neurons[2]._weights += np.array(
            [self.neurons[0].forward(np.array(x)),
             self.neurons[1].forward(np.array(x))]) * d_predicted_output
        self.neurons[2].bias += np.sum(d_predicted_output)


inputs = [([0, 0]),
          ([0, 1]),
          ([1, 0]),
          ([1, 1])]
expected_output = [0, 1, 1, 0]

model = Model()
# 5. Реализовать тренировочный цикл следующего вида:
epochs = 10000
for _ in range(epochs):
    for i in range(len(inputs)):
        x = inputs[i]
        y = expected_output[i]
        # прямой проход
        predicted_output = model.forward(x)
        # обратный проход
        error = loss(y, predicted_output)
        model.backward(x, error)

for x in inputs:
    predicted_output = model.forward(x)
    print(f"Результат работы нейронной сети для  {x}  после {epochs} эпох: ", round(predicted_output))
