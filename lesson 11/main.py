# Семинар 12 (в git 11) от 21.11.23
# лабу сделать как предыдущее только через torch
# Задания
# 1. Написать SimpleModel на другом уровне абстракции. Использовать model = nn.Sequential()
import torch
import torch.nn as nn

model = nn.Sequential(  # по сути nn.Sequential уже готовый класс
    nn.Linear(256, 64),  # 1й полносвязный слой
    nn.ReLU(),  # функция активации
    nn.Linear(64, 16),
    nn.Tanh(),  # функция активации
    nn.Linear(16, 4),
    nn.Softmax(dim=1)  # функция активации
)

print(model)


# 2. С помощью библиотеки torch реализовать модель с прямым проходом, состоящую из 3 полносвязных слоёв с функциями потерь: ReLU, tanh, Softmax.
# Длины векторов на входе 256, на выходе 4, промежуточные: 64 и 16. Использовать модули - nn.Module
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 4)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.tanh(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x


print(Model())


# 3. Реализовать модель с прямым проходом, состоящую из 2 свёрток (Conv) с функциями активации ReLU и 2 функций MaxPool.
# Первый слой переводит из 3 каналов в 8, второй из 8 слоёв в 16. На вход подаётся изображение размера 19х19. (19х19x3 -> 18x18x8 -> 9x9x8 -> 8x8x16 -> 4x4x16).
# Использовать модули - nn.Module
class MyConvModel(nn.Module):
    def __init__(self):
        super(MyConvModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=2)  # 1й сверточный слой
        self.conv2 = nn.Conv2d(8, 16, kernel_size=2)
        self.relu = nn.ReLU()  # функция активации
        self.max_pool1 = nn.MaxPool2d(kernel_size=2)  # функция пулинга
        self.max_pool2 = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = self.max_pool2(x)
        return x


print(MyConvModel())


# 4. Объединить сети из п.2 и п.3. На выход изображение размера 19х19, на выходе вектор из 4 элементов
class CombinedModel(nn.Module):
    def __init__(self):
        super(CombinedModel, self).__init__()
        self.conv_model = MyConvModel()
        self.model = Model()

    def forward(self, x):
        x = self.conv_model(x)
        x = x.view(-1, 256)
        x = self.model(x)
        return x


print(CombinedModel())
# пример работы
img = torch.Tensor(3, 19, 19)
print(img.shape)
new_model = CombinedModel()
res = new_model(img)
print(res)
