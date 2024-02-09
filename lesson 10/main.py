# Семинар 10 (в git 10_1) от 7.11.23
# Задание
# 1. Рассчитать average_digit (матрицу весов) для каждой цифры от 0 до 9, по аналогии с (avg_eight).
# 2. Объединить получившиеся веса в одну модель, которая на вход принимает картинку, а выдаёт вектор размера 10.
# 3. Рассчитать точность получившейся модели на тестовом наборе.
# 4. Визуализировать набор необработанных данных с помощью алгоритма t-SNE. Взять 30 изображений каждого класса, каждое изображение перевести в вектор размера (784), визуализировать полученные вектора с помощью t-SNE.
# 5. Визуализировать результаты работы вашей модели (эмбединги) с помощью алгоритма t-SNE. Прогнать изображения через вашу модель, получившиеся вектора размера (10) визуализировать с помощью t-SNE.
import numpy as np
import matplotlib.pyplot as plt
import torchvision

# Initializing the transform for the dataset
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5), (0.5))
])

# Downloading the MNIST dataset
train_dataset = torchvision.datasets.MNIST(
    root="./MNIST/train", train=True,
    transform=torchvision.transforms.ToTensor(),
    download=False)

test_dataset = torchvision.datasets.MNIST(
    root="./MNIST/test", train=False,
    transform=torchvision.transforms.ToTensor(),
    download=False)


def encode_label(j):
    # 5 -> [[0], [0], [0], [0], [0], [1], [0], [0], [0], [0]]
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


def shape_data(data):
    features = [np.reshape(x[0][0].numpy(), (784, 1)) for x in data]
    labels = [encode_label(y[1]) for y in data]
    return zip(features, labels)


train = list(shape_data(train_dataset))
test = list(shape_data(test_dataset))


# 1. Рассчитать average_digit (матрицу весов) для каждой цифры от 0 до 9, по аналогии с (avg_eight).
def average_digit(data, digit):
    filtered_data = [x[0] for x in data if np.argmax(x[1]) == digit]
    filtered_array = np.asarray(filtered_data)
    return np.average(filtered_array, axis=0)


avg_digits = []
for i in range(10):
    avg_digits.append(average_digit(train, i))


# 2. Объединить получившиеся веса в одну модель, которая на вход принимает картинку, а выдаёт вектор размера 10.
def sigmoid(x):  # вычисление вероятности
    return 1.0 / (1.0 + np.exp(-x))


def predict(x, w, b):  # предсказание с использованием матрицы весов и смещения
    return sigmoid(np.dot(w, x) + b)


# модель, которая принимает изображение, матрицу весов и смещение, и возвращает предсказанный вектор вероятностей для каждой цифры
def model(x, matrix, b):
    model_predict = []
    for i in range(10):
        W = np.transpose(matrix[i])
        model_predict.append(predict(x, W, b[i])[0][0])
    return model_predict


# 3. Рассчитать точность получившейся модели на тестовом наборе.
b = [-53, -34, -41, -44, -39, -37, -42, -38, -45.5, -41]  # массив смещений для каждой цифры
vectors = []
labels = []


def accuracy(test, avg_digits, mb):
    correct = 0
    for x, y in test:
        preds = model(x, avg_digits, mb)
        if np.argmax(preds) == np.argmax(y) and preds.count(1) < 2:
            vectors.append(model(x, avg_digits, mb))
            labels.append(np.argmax(y))
            correct += 1
    return correct / len(test)


print('Точность модели:', accuracy(test, avg_digits, b) * 100, '%')

# 4. Визуализировать набор необработанных данных с помощью алгоритма t-SNE. Взять 30 изображений каждого класса, каждое изображение перевести в вектор размера (784), визуализировать полученные вектора с помощью t-SNE.
from sklearn.manifold import TSNE

images_train = train_dataset.data.numpy()
labels_train = train_dataset.targets.numpy()

selected_images = []
selected_labels = []
for class_label in range(10):
    class_indexes = np.where(labels_train == class_label)[0]
    selected_indexes = np.random.choice(class_indexes, size=30, replace=False)  # выбираем случайные индексы изображений
    # выбираем 30 изображений из каждого класса
    selected_images.append(images_train[selected_indexes])
    selected_labels.append(labels_train[selected_indexes])
# объединяем выбранные изображения и метки в один массив
selected_images = np.concatenate(selected_images)
selected_labels = np.concatenate(selected_labels)
# изменение формы массива
images_vectors = selected_images.reshape(selected_images.shape[0], -1)
# получение вложенных векторов
tsne = TSNE(n_components=2, random_state=42)
embedded_vectors_train = tsne.fit_transform(images_vectors)

plt.scatter(embedded_vectors_train[:, 0], embedded_vectors_train[:, 1], c=selected_labels, cmap='tab10')
plt.title('Samples from Training Data')
plt.colorbar()
plt.show()

# 5. Визуализировать результаты работы вашей модели (эмбединги) с помощью алгоритма t-SNE. Прогнать изображения через вашу модель, получившиеся вектора размера (10) визуализировать с помощью t-SNE.
embedded_vectors_test = tsne.fit_transform(np.array(vectors))

plt.scatter(embedded_vectors_test[:, 0], embedded_vectors_test[:, 1], c=np.array(labels), cmap="tab10")
plt.title('Results on Training Data')
plt.colorbar()
plt.show()
