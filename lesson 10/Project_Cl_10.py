import numpy as np
import matplotlib.pyplot as plt
import torchvision

plt.figure(figsize=(6, 4))

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize((0.5), (0.5))])

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
    # print('features\n', len(features[0]))
    labels = [encode_label(y[1]) for y in data]
    # print('labels\n', len(labels[0]))
    return zip(features, labels)


train = list(shape_data(train_dataset))
test = list(shape_data(test_dataset))


def average_digit(data, digit):
    filtered_data = [x[0] for x in data if np.argmax(x[1]) == digit]
    filtered_array = np.asarray(filtered_data)
    return np.average(filtered_array, axis=0)


# Задание 1 #
avg_digits = []
for i in range(10):
    avg_digits.append(average_digit(train, i))


# Задание 2 #
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def predict(x, W, b):
    return sigmoid(np.dot(W, x) + b)


def model_p_to_v(xch, mx, mb):
    mpred = []
    for i in range(10):
        W = np.transpose(mx[i])
        mpred.append(predict(xch, W, mb[i])[0][0])
    return mpred


mb = [-53, -34, -41, -44, -39, -37, -42, -38, -45.5, -41]
# Пример #
# avg_eight = average_digit(test, 4)
# mpred_avg_eight = model_p_to_v(avg_eight, avg_digits, mb)
# print_model_pred(mpred_avg_eight)

# Задание 3 #
vectors_5 = []
labels_5 = []


def accuracy(test, avg_digits, mb):
    correct = 0
    total = len(test)
    for x, y in test:
        preds = model_p_to_v(x, avg_digits, mb)
        if np.argmax(preds) == np.argmax(y) and preds.count(1) < 2:
            vectors_5.append(model_p_to_v(x, avg_digits, mb))
            labels_5.append(np.argmax(y))
            correct += 1
    return correct / total


print('Точность модели:')
print(accuracy(test, avg_digits, mb))

# Задание 4 #
from sklearn.manifold import TSNE

images = train_dataset.data.numpy()
labels = train_dataset.targets.numpy()

selected_images = []
selected_labels = []
for class_label in range(10):
    class_indices = np.where(labels == class_label)[0]
    selected_indices = np.random.choice(class_indices, size=30, replace=False)
    selected_images.append(images[selected_indices])
    selected_labels.append(labels[selected_indices])
selected_images = np.concatenate(selected_images)
selected_labels = np.concatenate(selected_labels)

vectors = selected_images.reshape(selected_images.shape[0], -1)

tsne = TSNE(n_components=2, random_state=42)
embedded_vectors = tsne.fit_transform(vectors)

plt.scatter(embedded_vectors[:, 0], embedded_vectors[:, 1], c=selected_labels, cmap='tab10_r')
plt.title('Samples from Training Data')
plt.colorbar()
plt.show()

# Задание 5 #    
embedded_vectors_5 = tsne.fit_transform(np.array(vectors_5))

plt.scatter(embedded_vectors_5[:, 0], embedded_vectors_5[:, 1], c=np.array(labels_5), cmap="tab10_r")
plt.colorbar()
plt.title('Results from Training Data')
plt.show()
