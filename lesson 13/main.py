# Семинар 13 от 5.12.23
# Задание
# 1. Подготовьте данные для word2vec по одной из недавно прочитанных книг, удалив все символы, кроме букв и пробелов и обучите модель. Посмотрите результат.
# Обучение модели
import codecs
import numpy as np
import gensim
import re

with codecs.open('Rouling_Garri-Potter-narodnyy-perevod-_1_Garri-Potter-i-filosofskiy-kamen_RuLit_Me.txt',
                 encoding='utf-8', mode='r') as f:
    docs = f.readlines()

for i, line in enumerate(docs):
    filtered_line = re.sub('[^a-zA-Zа-яА-ЯёЁ ]+', ' ', line)
    docs[i] = filtered_line

# max_sentence_len = 12
# sentences = [sent for doc in docs for sent in doc.split('.')]
# sentences = [[word for word in sent.lower().split()[:max_sentence_len]] for sent in sentences]
sentences = [[word for word in sent.lower().split()] for sent in docs]
print(len(sentences), 'предложений')
# 16168 предложений

word_model = gensim.models.Word2Vec(sentences, vector_size=100, min_count=1, window=5, epochs=100)
pretrained_weights = word_model.wv.vectors
vocab_size, embedding_size = pretrained_weights.shape
print(vocab_size, embedding_size)
# 21571 100
print('Похожие слова:')
for word in ['alohomora', 'locomotor', 'палочка']:
    most_similar = ', '.join('%s (%.2f)' % (similar, dist) for similar, dist in word_model.wv.most_similar(word)[:8])
    print('  %s -> %s' % (word, most_similar))

# print(word_model.wv.get_vector('гарри'))
# vec = word_model.wv.get_vector('гарри') - word_model.wv.get_vector('поттер') + word_model.wv.get_vector('мальчик')
# print(word_model.wv.similar_by_vector(vec), sep='\n')

# 3. Из этого же текста (п.1) возьмите небольшой фрагмент, разбейте на предложения с одинаковым числом символов. Каждый символ предложения закодируйте с помощью one hot encoding. В итоге у вас должен получиться массив размера (n_sentences, sentence_len, encoding_size).
from scipy import stats as st

with codecs.open('Rouling_Garri-Potter-narodnyy-perevod-_1_Garri-Potter-i-filosofskiy-kamen_RuLit_Me.txt',
                 encoding='utf-8', mode='r') as f:
    docs_part = f.readlines()
# print(docs_part[:100])
sentences_part = [[word for word in sent.lower().split()] for sent in docs]
# print(sentences_part[:100])
sentences_part = [sent for sent in sentences_part if len(sent) != 0]

sentence_len = 20  # длина предложения
sentences_part_same = [sent for sent in sentences_part if
                       sum(map(len, sent)) == sentence_len]  # разбиваем на предложения с одинаковым числом символов
sentences_part_same = np.array([list(''.join(sent)) for sent in sentences_part_same])
# print(sentences_part_same)
unique_letters = np.unique(sentences_part_same)
alphabet = {}
for i, letter in enumerate(unique_letters):
    arr = [0] * len(unique_letters)
    arr[i] = 1
    alphabet[letter] = arr
print(*alphabet)

n_sentences = np.ndarray.tolist(sentences_part_same)

for i in range(len(n_sentences)):
    for j in range(len(n_sentences[0])):
        n_sentences[i][j] = alphabet[n_sentences[i][j]]

res = np.array(n_sentences)
print(res.shape)
print(res[:5, :, :20])


# 2. Для обучения на нефтяных скважин добавьте во входные данные информацию со столбцов Gas, Water (т.е. размер x_data будет (440, 12, 3)) и обучите новую модель. Выход содержит Liquid, Gas и Water (для дальнейшего предсказания). Графики с результатами только для Liquid.
print('\n>>>>>>>>>>>>>>>>>>>>>>>>>>> задание 2 >>>>>>>>>>>>>>>>>>>>>>>>>>>\n')
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

df = pd.read_csv('production.csv')
# print(df.head())
# Подготовка данных по добыче
liquid = df.groupby(['API'])[['Liquid', 'Gas', 'Water']].apply(lambda df_: df_.reset_index(drop=True))
# print(liquid.head())
# Масштабирование и деление на трейн/тест
liquid['Liquid'] = liquid['Liquid'] / liquid['Liquid'].max()
liquid['Gas'] = liquid['Gas'] / liquid['Gas'].max()
liquid['Water'] = liquid['Water'] / liquid['Water'].max()

# print(liquid.iloc[0])
data = liquid.to_numpy()
data = data.reshape((50, 24, 3))

data_tr = data[:40]
data_tst = data[40:]
# print(data_tr.shape, data_tst.shape)

x_data = [data_tr[:, i:i + 12] for i in range(11)]
y_data = [data_tr[:, i + 1:i + 13] for i in range(11)]

x_data = np.concatenate(x_data, axis=0)
y_data = np.concatenate(y_data, axis=0)
# print(x_data.shape, y_data.shape)

tensor_x = torch.Tensor(x_data)  # transform to torch tensor
tensor_y = torch.Tensor(y_data)

oil_dataset = TensorDataset(tensor_x, tensor_y)  # create your datset
oil_dataloader = DataLoader(oil_dataset, batch_size=16)  # create your dataloader

for x_t, y_t in oil_dataloader:
    break


# print(x_t.shape, y_t.shape)


class OilModel(nn.Module):
    def __init__(self, timesteps=12, units=32):
        super().__init__()
        self.lstm1 = nn.LSTM(3, units, 2, batch_first=True)
        self.dense = nn.Linear(units, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        h, _ = self.lstm1(x)
        outs = []
        for i in range(h.shape[0]):
            outs.append(self.relu(self.dense(h[i])))
        # print(outs)
        out = torch.stack(outs, dim=0)
        return out


model = OilModel()
opt = optim.Adam(model.parameters())
criterion = nn.MSELoss()
NUM_EPOCHS = 100

for epoch in range(NUM_EPOCHS):  # loop over the dataset multiple times

    running_loss = 0.0
    num = 0
    for x_t, y_t in oil_dataloader:
        # zero the parameter gradients
        opt.zero_grad()

        # forward + backward + optimize
        outputs = model(x_t)
        loss = criterion(outputs, y_t)
        loss.backward()
        opt.step()

        # вывод статистики
        running_loss += loss.item()
        num += 1

    print(f'[Эпоха: {epoch + 1:3d}] loss: {running_loss / num:.3f}')
print('Обучение закончено\n')

# Предскажем на год вперёд используя данные только первого года
x_tst = data_tst[:, :12]
predicts = np.zeros((x_tst.shape[0], 0, x_tst.shape[2]))

for i in range(12):
    x = np.concatenate((x_tst[:, i:], predicts), axis=1)
    x_t = torch.from_numpy(x).float()
    pred = model(x_t).detach().numpy()
    last_pred = pred[:, -1:]  # нас интересует только последний месяц
    predicts = np.concatenate((predicts, last_pred), axis=1)
# print(predicts)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
for iapi in range(4):
    plt.subplot(2, 2, iapi + 1)
    plt.plot(np.arange(x_tst.shape[1]), x_tst[iapi, :, 0], label='Текущее')
    plt.plot(np.arange(predicts.shape[1]) + x_tst.shape[1], predicts[iapi, :, 0], label='Предсказание')
    plt.legend()
plt.show()
