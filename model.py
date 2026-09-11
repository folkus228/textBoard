import torch.nn as nn
import torch.nn.functional as F


class CRNN(nn.Module):
    def __init__(self, vocab_size:int, drop_rate:float=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 7, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 64, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 96, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(96),

            nn.Conv2d(96, 128, 7, padding=3),
            nn.ReLU(),
            nn.MaxPool2d((3, 1), (3, 1)),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 256, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.BatchNorm2d(256),

            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((1, None))
        )
        self.drop = nn.Dropout1d(p=drop_rate)
        self.bilstm = nn.LSTM(256, 256, batch_first=True, bidirectional=True)
        self.out_lin = nn.Linear(512, vocab_size)
    def forward(self, x):
        x = self.conv(x)
        x = x.squeeze(2).transpose(2, 1) # [B, W, C]
        x, _ = self.bilstm(self.drop(x))
        x = self.out_lin(x)
        x = F.log_softmax(x, dim=2)
        return x
