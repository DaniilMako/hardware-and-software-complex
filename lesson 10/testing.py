import torch
import torch.nn as nn
import torchvision
import numpy as np
import matplotlib.pyplot as plt

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.MNIST(
    root="./MNIST/train", train=True,
    transform=transform,
    download=True)

test_dataset = torchvision.datasets.MNIST(
    root="./MNIST/test", train=False,
    transform=transform,
    download=True)

weight_matrices = []
for digit in range(10):
    digit_data = [x[0].numpy().flatten() for x in train_dataset if x[1] == digit]
    digit_data = np.stack(digit_data)
    weight_matrix = np.mean(digit_data, axis=0)
    weight_matrices.append(weight_matrix)


class DigitClassifier(nn.Module):
    def __init__(self):
        super(DigitClassifier, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)

        # TODO:x = x.flatten()

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = DigitClassifier()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_list = []


def train_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            loss_list.append(loss.item())
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}")


def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total * 100
    return accuracy


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

num_epochs = 10
train_model(model, train_loader, criterion, optimizer, num_epochs)
accuracy = test_model(model, test_loader)
print(f"Accuracy: {accuracy:.2f}%")

torch.save(model.state_dict(), "C:\\Users\\lilar\\PycharmProjects\\models\\best.pt")
plt.plot(loss_list)
plt.show()

# vectors = []
# labels = []
# for digit in range(10):
#     digit_data = [x[0].numpy().flatten() for x in train_dataset if x[1] == digit]
#     digit_data = np.stack(digit_data)
#     vectors.append(digit_data[:30])
#     labels.extend([digit] * 30)
# vectors = np.concatenate(vectors)
#
# tsne = TSNE(n_components=2, random_state=42)
# vectors_tsne = tsne.fit_transform(vectors)
#
# plt.figure(figsize=(10, 8))
# plt.scatter(vectors_tsne[:, 0], vectors_tsne[:, 1], c=labels, cmap='viridis')
# plt.colorbar()
# plt.title("t-SNE Visualization of MNIST Dataset")
# plt.xlabel("Dimension 1")
# plt.ylabel("Dimension 2")
# plt.show()

raw_vectors = []
for digit in range(10):
    digit_data = [x[0].numpy().flatten() for x in train_dataset if x[1] == digit][:30]
    digit_data = np.stack(digit_data)
    raw_vectors.extend(digit_data)

raw_vectors = np.array(raw_vectors)

from sklearn.manifold import TSNE

tsne = TSNE(n_components=2)
raw_embeddings = tsne.fit_transform(raw_vectors)

plt.scatter(raw_embeddings[:, 0], raw_embeddings[:, 1])
plt.show()

model_vectors = []
for images, _ in test_loader:
    images = images.to(device)
    outputs = model(images)
    embeddings = outputs.detach().cpu().numpy()
    model_vectors.extend(embeddings)

model_vectors = np.array(model_vectors)

tsne = TSNE(n_components=2)
model_embeddings = tsne.fit_transform(model_vectors)

plt.scatter(model_embeddings[:, 0], model_embeddings[:, 1])
plt.show()
