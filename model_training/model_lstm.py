import torch
from torch import nn

class CheaterClassifier(nn.Module):
    def __init__(self, embedding_dim=5, lstm_layers=3, lstm_hidden_dim=128):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(13, self.embedding_dim)
        self.batchNorm = nn.BatchNorm1d(128)
        self.lstm_layers = lstm_layers
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm = nn.LSTM(128*self.embedding_dim, self.lstm_hidden_dim, self.lstm_layers)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.lstm_hidden_dim, 128)
        self.fc2 = nn.Linear(128, 1)


    def forward(self, x):
        x = self.embedding(x)
        x = self.batchNorm(x)
        x = self.flatten(x)
        x, (h, c) = self.lstm(x)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.sigmoid(self.fc2(x))
        return x
