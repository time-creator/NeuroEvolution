import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO: definitely not an autoencoder; for now just a simple MNIST NN
class EvolutionNet(nn.Module):

    def __init__(self):
        super(EvolutionNet, self).__init__()

        self.fc1 = nn.Linear(28 * 28, 400)
        self.fc2 = nn.Linear(400, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), 0)
        return x

class MNISTTwoLayer(nn.Module):

    def __init__(self):
        super(EvolutionNet, self).__init__()

        self.fc1 = nn.Linear(28 * 28, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), 0)
        return x
