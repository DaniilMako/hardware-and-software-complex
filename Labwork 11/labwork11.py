# Лабораторная работа 11.
# 1. С помощью библиотеки torch реализовать модель с прямым проходом, состоящую из 3 полносвязных слоёв с функциями активации: Sigmoid, tanh, Softmax.
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # Длины векторов на входе 256, на выходе 4, промежуточные: 64 и 16. Использовать barebone подход.
        self.fc1 = nn.Parameter(torch.randn(256, 64))
        self.fc2 = nn.Parameter(torch.randn(64, 16))
        self.fc3 = nn.Parameter(torch.randn(16, 4))

    def forward(self, x):
        # функции активации:
        x = torch.sigmoid(x @ self.fc1)
        x = torch.tanh(x @ self.fc2)
        x = F.softmax(x @ self.fc3, dim=1)
        return x


model = Model()

input_vector = torch.randn(1, 256)
output_vector = model(input_vector)

print(output_vector)
