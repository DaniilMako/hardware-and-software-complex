# Лабораторная работа 12.
# 1. Задача классификации изображений. Повторить тренировку модели (train) и запустить классификацию изображений (inference).
# >>>>>>>>>>>>>>>>>> Загрузите и нормализуйте обучающие и тестовые наборы данных CIFAR10 с помощью torchvision
import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=True, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

import matplotlib.pyplot as plt
import numpy as np


# функции для отображения изображения


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# получение нескольких рандомных изображений
dataiter = iter(trainloader)
images, labels = next(dataiter)

# показать/вывести изображения
imshow(torchvision.utils.make_grid(images))
# вывести метки
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)), '\n')

# >>>>>>>>>>>>>>>>>> Определение сверточной нейронной сети
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 5, padding=2)  # 1й слой свёртки
        self.batch1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 5, padding=2)  # 2й слой свёртки
        self.batch2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 5, padding=2)  # 3й слой свёртки
        self.batch3 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)  # слой пулинга
        self.fc1 = nn.Linear(64 * 4 * 4, 512)  # 1й полносвязный слой
        self.fc2 = nn.Linear(512, 400)  # 2й полносвязный слой
        self.fc3 = nn.Linear(400, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 32)
        self.fc7 = nn.Linear(32, 10)

    def forward(self, x):
        # x = self.pool(F.relu(self.batch1(self.conv1(x))))
        # x = self.pool(F.relu(self.batch2(self.conv2(x))))
        # x = self.pool(F.relu(self.batch3(self.conv3(x))))
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)  # преобразование в вектор
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = self.fc7(x)
        return x
    # x = torch.flatten(x, 1)  # преобразует многомерный тензор в одномерный тензор


net = Net()

# >>>>>>>>>>>>>>>>>> Определение функции потерь
import torch.optim as optim

criterion = nn.CrossEntropyLoss()  # функция потерь
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  # обновление весов
optimizer = optim.Adam(net.parameters(), lr=0.0001, betas=(0.9, 0.999))  # обновление весов

# >>>>>>>>>>>>>>>>>> Обучите сеть на основе обучающих данных
for epoch in range(10):  # повторение цикла по набору данных несколько раз

    running_loss = 0.0  # для статистики
    for i, data in enumerate(trainloader, 0):
        # получаем inputs: data - это список из [inputs, labels]
        inputs, labels = data

        # обнуление градиентов параметров
        optimizer.zero_grad()

        # forward + backward + optimize (прямой проход + обратный проход + оптимизация)
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # вывод статистики
        running_loss += loss.item()
        if i % 2000 == 1999:  # вывод каждых 2000 мини-батчей (mini-batches)
            print(f'[{epoch + 1}, {i + 1:5d}]   loss: {running_loss / 2000:.3f}')
            running_loss = 0.0
print('Обучение закончено\n')

# сохранение нейронки после обучения
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)

# >>>>>>>>>>>>>>>>>> Протестируйте сеть на тестовых данных
dataiter = iter(testloader)
images, labels = next(dataiter)

# вывод изображений и меток
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))
# загрузка нейронки
net = Net()
net.load_state_dict(torch.load(PATH))

outputs = net(images)

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(4)), '\n')

correct = 0
total = 0
# поскольку мы не тренируемся, нам не нужно вычислять градиенты для наших выходных данных
with torch.no_grad():
    for data in testloader:
        images, labels = data
        # вычисление выходных данных путем прогона изображений по сети
        outputs = net(images)
        # класс с наибольшей энергией - это то, что мы выбираем в качестве прогноза
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Доля правильных ответов сети на 10000 тестовых изображениях: {100 * correct // total} %\n')

# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1

# 2. Получить максимальную точность классификации (минимальный loss) путём изменения модели, например, добавлением скрытых слоёв.
# вывод точности для каждого класса
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Доля правильных ответов для класса: {classname:5s} is {accuracy:.1f} %')
