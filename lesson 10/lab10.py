import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torchvision

# Initializing the transform for the dataset
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5), (0.5))
])
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
    e = np.zeros((10,1))
    e[j] = 1.0
    return e

def decode_label(e):
    return np.argmax(e)

def shape_data(data):
    features = [np.reshape(x[0][0].numpy(), (784,1)) for x in data]
    #print('features\n', len(features[0]))
    labels = [encode_label(y[1]) for y in data]
    #print('labels\n', len(labels[0]))
    return zip(features, labels)

train = shape_data(train_dataset)
test = shape_data(test_dataset)

train = list(train)
test = list(test)


def average_digit(data, digit):
    filtered_data = [x[0] for x in data if np.argmax(x[1]) == digit]
    filtered_array = np.asarray(filtered_data)
    return np.average(filtered_array, axis=0)

avg = [average_digit(train, x) for x in range(10)]


def predict(test, avg):
    test = [x[0] for x in test]
    dot_digits = []
    for x in test:
        x = np.transpose(x)
        dot_digits.append([np.dot(x, avg_x) for avg_x in avg])
    answer = np.argmax(np.array(dot_digits), axis=1)
    encoded_labels = np.zeros((answer.shape[0], 10, 1))
    for i, sub_arr in enumerate(answer):
        encoded_labels[i][sub_arr[0]] = 1.0
    return encoded_labels, dot_digits


def calculate_accuracy(labels, predicted):
    total = len(labels)
    correct_answers = 0
    for i in range(total):
        if np.array_equal(labels[i], predicted[i]):
            correct_answers += 1
    return correct_answers * 100 / total


def tsne_plot(x_subset, y_subset, name):
    tsne = TSNE(random_state=42, n_components=2, verbose=0, perplexity=5).fit_transform(x_subset)
    plt.scatter(tsne[:, 0], tsne[:, 1], s=5, c=y_subset, cmap='Spectral')
    plt.gca().set_aspect('equal', 'datalim')
    plt.colorbar(boundaries=np.arange(11) - 0.5).set_ticks(np.arange(10))
    plt.title(name, fontsize=20)
    plt.show()

predicted, answers = predict(test, avg)
labels = [encode_label(y[1]) for y in test_dataset]
print(calculate_accuracy(labels, predicted))

test_ans = np.array([np.argmax(x) for x in labels])
answers = np.array(answers)
n_samples, nx, ny, nz = answers.shape
answers = answers.reshape(n_samples, nx)
tsne_plot(answers, test_ans, 'Samples from training data')

x = np.array([np.reshape(x[0][0].numpy(), (784,)) for x in train_dataset])
y = np.array([y[1] for y in train_dataset])
x_subset = x[:3000]
y_subset = y[:3000]
tsne_plot(x_subset, y_subset, 'Results on training data')
