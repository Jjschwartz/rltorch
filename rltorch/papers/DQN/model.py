"""The DQN Model """
import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):

    def __init__(self, num_actions):
        super().__init__()
        # input = 84 x 84 x 4 image
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        # Width = Height = ((84-8)/4)+1 = 20, channels = 32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        # Width = Height = ((20-4)/2)+1 = 9, channels = 64
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # Width = Height = ((9-3)/1)+1 = 7, channels = 64
        self.fc1 = nn.Linear(64*7*7, 512)
        # neurons = 512
        self.out = nn.Linear(512, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.out(x)
        return x

    def save_DQN(self, file_path):
        torch.save(self.state_dict(), file_path)

    def load_DQN(self, file_path, device='cpu'):
        self.load_state_dict(torch.load(file_path,
                                        map_location=torch.device(device)))

    def get_action(self, x):
        with torch.no_grad():
            return self.forward(x).max(1)[1]
