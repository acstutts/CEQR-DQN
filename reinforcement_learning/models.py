import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def init_weights(m, gain):
    if (type(m) == nn.Linear) | (type(m) == nn.Conv2d):
        nn.init.orthogonal_(m.weight, gain)
        nn.init.zeros_(m.bias)

class CNN_UADQN(nn.Module):

    def __init__(self, observation_space, n_outputs_1, obs_space_channel, weight_scale=np.sqrt(2)):

        super().__init__()
        self.weight_scale = weight_scale

        if len(observation_space.shape) != 3:
            raise NotImplementedError

        in_channels = obs_space_channel

        # One hidden 2D convolution layer:
        #   in_channels: variable
        #   out_channels: 16
        #   kernel_size: 3 of a 3x3 filter matrix
        #   stride: 1
        self.conv = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1)

        # Final fully connected hidden layer:
        #   the number of linear unit depends on the output of the conv
        #   the output consist 128 rectified units
        def size_linear_unit(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1
        num_linear_units = size_linear_unit(10) * size_linear_unit(10) * 16
        self.fc_hidden = nn.Linear(in_features=num_linear_units, out_features=128)

        # Output layer:
        self.output = nn.Linear(in_features=128, out_features=n_outputs_1)

        self.conv.apply(lambda x: init_weights(x, self.weight_scale))
        self.fc_hidden.apply(lambda x: init_weights(x, self.weight_scale))
        self.output.apply(lambda x: init_weights(x, self.weight_scale))

    # As per implementation instructions according to pytorch, the forward function should be overwritten by all
    # subclasses
    def forward(self, x):
        if len(x.shape) != 4:
            x = x.unsqueeze(0)

        x = x.permute(0, 3, 1, 2)

        # Rectified output from the first conv layer
        x = F.relu(self.conv(x))

        # Rectified output from the final hidden layer
        x = F.relu(self.fc_hidden(x.contiguous().view(x.size(0), -1)))

        # Returns the output from the fully-connected linear layer
        return self.output(x)
    
class CNN_CEQRDQN(nn.Module):

    def __init__(self, device, observation_space, n_quantiles, n_outputs_1, obs_space_channel, weight_scale=np.sqrt(2)):

        super().__init__()
        self.device = device
        self.n_quantiles = n_quantiles
        self.obs_space_channel = obs_space_channel
        self.n_actions = int(n_outputs_1 / n_quantiles)
        self.weight_scale = np.sqrt(2)

        if len(observation_space.shape) != 3:
            raise NotImplementedError

        self.conv = nn.Sequential(
            nn.Conv2d(self.obs_space_channel,16,3,stride=1,padding=1),
            nn.PReLU(num_parameters=16),
        )

        self.out_hidden = nn.Sequential(
            nn.Linear(in_features=1600, out_features=1600),
            nn.ReLU()
        )
        
        self.output = nn.Sequential(
            nn.Linear(1600, n_outputs_1),
        )

        self.NIG = nn.Sequential(
            nn.Linear(1600, 4*self.n_actions*2*self.n_quantiles), # = 4 * 2 * n_outputs_1
        )

        self.conv.apply(lambda x: init_weights(x, self.weight_scale))
        self.out_hidden.apply(lambda x: init_weights(x, self.weight_scale))
        self.output.apply(lambda x: init_weights(x, self.weight_scale))
        self.NIG.apply(lambda x: init_weights(x, self.weight_scale))


    def evi_split(self, out):
        mu, logv, logalpha, logbeta = torch.split(out, self.n_actions*2*self.n_quantiles, dim=-1)

        v = F.softplus(logv)
        alpha = F.softplus(logalpha) + 1
        beta = F.softplus(logbeta)
        return torch.concat([mu, v, alpha, beta], axis=-1)

    def forward(self, obs):
        if len(obs.shape) != 4:
            obs = obs.unsqueeze(0)
        obs = obs.permute(0,3,1,2)
        obs = self.conv(obs)
        obs = obs.contiguous().view(obs.size(0), -1)

        hidden = self.out_hidden(obs)
        output = self.output(hidden)
        G_evi = self.NIG(hidden)
        G_evi = self.evi_split(G_evi)

        return output, G_evi