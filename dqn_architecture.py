import torch
from torch import nn
import torch.nn.functional as F

class DQN(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_dim=256, enable_dueling=True):
        super(DQN, self).__init__()

        self.enable_dueling = enable_dueling
        self.fc1 = nn.Linear(state_dim, hidden_dim)

        if self.enable_dueling:
            # value branch
            self.fc_value = nn.Linear(hidden_dim, 256)
            self.value = nn.Linear(256, 1)
            # advantage branch
            self.fc_advantages = nn.Linear(hidden_dim, 256)
            self.advantages = nn.Linear(256, action_dim)
        else:
            # simple Q network
            self.output = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        latent = F.relu(self.fc1(x))

        if self.enable_dueling:
            # compute value and advantage then combine
            v = F.relu(self.fc_value(latent))
            V = self.value(v)
            a = F.relu(self.fc_advantages(latent))
            A = self.advantages(a)
            Q = V + A - torch.mean(A, dim=1, keepdim=True)
        else:
            Q = self.output(latent)

        return Q
