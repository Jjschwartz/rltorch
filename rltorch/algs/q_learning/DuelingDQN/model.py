"""The DQN Model """
import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingDQN(nn.Module):

    def __init__(self, input_dim, layers, num_actions, dueling_layers):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(input_dim[0], layers[0])])
        for l in range(1, len(layers)):
            self.layers.append(nn.Linear(layers[l-1], layers[l]))

        self.value_layers = nn.ModuleList(
            [nn.Linear(layers[-1], dueling_layers[0])]
        )

        self.adv_layers = nn.ModuleList(
            [nn.Linear(layers[-1], dueling_layers[0])]
        )

        for l in range(1, len(dueling_layers)):
            self.value_layers.append(
                nn.Linear(dueling_layers[l-1], dueling_layers[l])
            )
            self.adv_layers.append(
                nn.Linear(dueling_layers[l-1], dueling_layers[l])
            )

        self.value_out = nn.Linear(dueling_layers[-1], 1)
        self.adv_out = nn.Linear(dueling_layers[-1], num_actions)

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))

        v = x
        for v_layer in self.value_layers:
            v = F.relu(v_layer(v))
        v = self.value_out(v)

        adv = x
        for adv_layer in self.adv_layers:
            adv = F.relu(adv_layer(adv))
        adv = self.adv_out(adv)

        adv_avg = torch.mean(adv, dim=1, keepdim=True)
        q_vals = v + adv - adv_avg
        return q_vals

    def save_DQN(self, file_path):
        torch.save(self.state_dict(), file_path)

    def load_DQN(self, file_path):
        self.load_state_dict(torch.load(file_path))

    def get_action(self, x):
        with torch.no_grad():
            if len(x.shape) == 1:
                x = x.view(1, -1)
            return self.forward(x).max(1)[1]
