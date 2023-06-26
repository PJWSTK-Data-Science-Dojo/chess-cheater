import torch
from torch import nn

class CheaterClassifier(nn.Module):
    def __init__(self, embedding_dim=5, lstm_layers=3, lstm_hidden_dim=128, fc1=512, fc2=512, fc3=256, fc4=128, conv1=128, conv2=128, conv3=128):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(13, self.embedding_dim)
        self.batchNorm = nn.BatchNorm1d(128)
        self.drop = nn.Dropout(p=0.5)
        self.lstm_layers = lstm_layers
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm = nn.LSTM(128*self.embedding_dim, self.lstm_hidden_dim, self.lstm_layers)
        self.conv1 = nn.Conv1d(128, conv1, 3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(conv1, conv2, 3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(conv2, conv3, 3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(3, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.lstm_hidden_dim, 128)
        self.fc2 = nn.Linear(128, 1)
        self.fc3 = nn.Linear(fc2, fc3)
        self.fc4 = nn.Linear(fc3, fc4)
        self.fc5 = nn.Linear(fc4, 1)


    def forward(self, x):
        x = self.embedding(x)
        x = self.batchNorm(x)
        x = self.flatten(x)
        x, (h, c) = self.lstm(x)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.sigmoid(self.fc2(x))
        return x
