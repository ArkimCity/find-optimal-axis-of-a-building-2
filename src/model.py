from json import JSONEncoder

import torch
import torch.nn as nn
from torch.utils.data import Dataset


class EncodeTensor(JSONEncoder, Dataset):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu().detach().numpy().tolist()
        return super(EncodeTensor, self).default(obj)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 2)  # FIXME: result shape fix

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 32 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DirectionPredictionModelWithTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DirectionPredictionModelWithTransformer, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, nhead=2), num_layers=2
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = embedded.permute(
            1, 0, 2
        )  # Transformer expects shape: (seq_len, batch_size, input_size)
        out = self.transformer(embedded)
        out = self.fc(out[-1])  # Take the output of the last token
        return out
