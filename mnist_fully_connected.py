from keras.datasets import mnist
import torch
import numpy as np


class MNISTNet(torch.nn.Module):

    def __init__(self, n_hidden_neurons):
        super(MNISTNet, self).__init__()
        self.fc1 = torch.nn.Linear(side_size * side_size, n_hidden_neurons)
        self.act1 = torch.nn.Sigmoid()
        self.fc2 = torch.nn.Linear(n_hidden_neurons, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        return x


#параметры
n_train = 60000
n_test = 10000
side_size = 28

batch_size = 100
n_epohs = 100
n_hidden_neurons = 100

torch.manual_seed(42)

# скачиваем и подготавливаем данные
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = torch.tensor(x_train)
x_test = torch.tensor(x_test)
y_train = torch.tensor(y_train)
y_test = torch.tensor(y_test)

x_train = x_train.float()
x_test = x_test.float()

#преобразуем размерность данных 60000 * 28 * 28 в 60000 * 784
x_train = x_train.reshape([-1, side_size * side_size])
x_test = x_test.reshape([-1, side_size * side_size])

#инициализация нейронной сети
mnist_net = MNISTNet(n_hidden_neurons)

learning_rate = 0.001
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mnist_net.parameters(), lr = learning_rate)

#обучение
for _ in range(n_epohs):
    order = np.random.permutation(n_train)

    for start in range(0, n_train, batch_size):
        optimizer.zero_grad()
        x_batch = x_train[order[start : start + batch_size]]
        y_batch = y_train[order[start : start + batch_size]]

        batch_predictions = mnist_net.forward(x_batch)
        loss_value = loss(batch_predictions, y_batch)
        loss_value.backward()
        
        optimizer.step()
        
#проверка
predictions = mnist_net.forward(x_test)
accuracy = (predictions.argmax(dim = 1) == y_test).float().mean()
print(accuracy)
