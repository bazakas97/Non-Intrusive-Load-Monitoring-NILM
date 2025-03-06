import torch
import torch.nn as nn
import torch.nn.functional as F


class AdvancedSeq2PointCNN(nn.Module):
    """
    A deeper/more advanced CNN model for Seq2Point-style NILM.
    You can tweak channels, kernel sizes, etc. as desired.
    """

    def __init__(self, input_dim=1, output_dim=1, window_size=160):
        super().__init__()
        self.window_size = window_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Convs...
        self.conv1 = nn.Conv1d(in_channels=self.input_dim, out_channels=32, kernel_size=8)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=8)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=6)
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=6)
        self.conv5 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5)
        
        self.dropout = nn.Dropout(0.2)
        
        # Do a mock forward to find out flatten dimension:
        with torch.no_grad():
            x_test = torch.zeros((1, self.window_size, self.input_dim))  # batch=1
            x_test = x_test.permute(0, 2, 1)  # => [1, 1, 160]
            x_test = self.conv1(x_test)
            x_test = self.conv2(x_test)
            x_test = self.conv3(x_test)
            x_test = self.conv4(x_test)
            x_test = self.conv5(x_test)
            out_shape = x_test.shape  # e.g. [1, 128, 132]
            flattened_size = out_shape[1] * out_shape[2]
        
        self.fc1 = nn.Linear(flattened_size, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, self.output_dim)


    def forward(self, x):
        # x shape: [batch_size, window_size, input_dim]
        # We need [batch_size, input_dim, window_size] for Conv1d
        x = x.permute(0, 2, 1)  # => [batch_size, 1, window_size]

        x = F.relu(self.conv1(x))  # => [batch_size, 32, ...]
        x = F.relu(self.conv2(x))  # => [batch_size, 64, ...]
        x = F.relu(self.conv3(x))  # => [batch_size, 64, ...]
        x = F.relu(self.conv4(x))  # => [batch_size, 128, ...]
        x = F.relu(self.conv5(x))  # => [batch_size, 128, ...]

        x = self.dropout(x)
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
