import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import init
from torch.nn.parameter import Parameter
from torch.distributions.categorical import Categorical


def MLP(sizes, activation, dropped:bool=False, drop_rate=0.2, out_activation=nn.Identity):
    layers = []
    for i in range(len(sizes)-1):
        act = activation if i < len(sizes)-2 else out_activation
        if dropped:
            layers += [nn.Linear(sizes[i], sizes[i+1], act(), nn.Dropout(drop_rate))]
        else:
            layers += [nn.Linear(sizes[i], sizes[i+1]), act()]
    return nn.Sequential(*layers)


class GNNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weights = Parameter(torch.FloatTensor(in_dim, out_dim))
        self.init_weights()

    def init_weights(self):
        init.xavier_uniform(self.weights)

    # A_adjust = D^(-0.5)*(A+I)*D^(0.5)
    def forward(self, A_adjust, H):
        support = torch.matmul(H, self.weights)
        out = torch.matmul(A_adjust, support)
        return out


class GNN(nn.Module):
    def __init__(self, node_num, feature_dim, hidden_size, layers):
        super().__init__()
        self.node_num = node_num
        self.feature_dim = feature_dim

        self.GNN_layers = []
        for i in range(layers):
            if i == 0:
                self.GNN_layers.append(GNNLayer(feature_dim, hidden_size))
            elif i == layers-1:
                self.GNN_layers.append(GNNLayer(hidden_size, feature_dim))
            else:
                self.GNN_layers.append(GNNLayer(hidden_size, hidden_size))
        self.GNN_layers = nn.ModuleList(self.GNN_layers)

    # node_num: n
    # node_features: batch_size*n*feature_num
    # node_adjacency: batch_size*n*n
    # obs: batch_size*n*(feature_num+n)
    def forward(self, obs):
        if len(obs.size()) == 3:    # batched
            A_adjust, H = torch.split(obs, [self.node_num, self.feature_dim], dim=2)
        else: 
            A_adjust, H = torch.split(obs, [self.node_num, self.feature_dim], dim=1)

        for layer in self.GNN_layers:
            H = F.relu(layer(A_adjust, H))
        
        if len(H.size()) == 3:
            embedding = torch.flatten(H, start_dim=1)
        else:
            embedding = torch.flatten(H)
        return embedding


class GNNActor(nn.Module):
    def __init__(self, GNN, node_num, feature_dim, hidden_sizes, actions, activation):
        super().__init__()
        self.GNN = GNN
        self.policy_function = MLP([node_num*feature_dim]+list(hidden_sizes)+[actions], activation)
    
    def get_policy(self, obs, logits:bool=False):   # logits is the log probability, log_p = ln(p)
        embedding = self.GNN(obs)
        log_prob = self.policy_function(embedding)
        if logits:
            policy = log_prob
        else:
            policy = Categorical(logits=log_prob)
        return policy

    def forward(self, obs, act=None):
        pi = self.get_policy(obs, logits=False)
        log_a = None
        if act is not None:
            log_a = pi.log_prob(act)
        return pi, log_a


class GNNCritic(nn.Module): 
    def __init__(self, GNN, node_num, feature_dim, hidden_sizes, activation):
        super().__init__()  
        self.GNN = GNN
        self.value_function = MLP([node_num*feature_dim]+list(hidden_sizes)+[1], activation)
    
    def forward(self, obs):
        embedding = self.GNN(obs)
        value = torch.squeeze(self.value_function(embedding), -1)
        return value        


class GNNActorCritic(nn.Module):
    def __init__(self, obs_space, act_space, GNN_layers=2, GNN_hidden_size=256, MLP_hidden_sizes=(256, 256), MLP_activation=nn.ReLU):
        super().__init__()

        node_num = obs_space.shape[0]
        feature_dim = obs_space.shape[1]-node_num
        action_dim = act_space.n

        self.GNN = GNN(node_num, feature_dim, GNN_hidden_size, GNN_layers)
        self.actor = GNNActor(self.GNN, node_num, feature_dim, MLP_hidden_sizes, action_dim, MLP_activation)
        self.critic = GNNCritic(self.GNN, node_num, feature_dim, MLP_hidden_sizes, MLP_activation)

    def step(self, obs, mask):
        with torch.no_grad():
            pi = self.actor.get_policy(obs, logits=False)
            log_pi = self.actor.get_policy(obs, logits=True)
            
            log_pi_delta = torch.zeros(mask.size())
            log_pi_delta[mask==0] = torch.tensor(float('-Inf'))
            log_pi += log_pi_delta
            mask_pi = Categorical(logits=log_pi)

            a = mask_pi.sample()
            log_a = pi.log_prob(a)

            v = self.critic(obs)
        return a.cpu().numpy(), v.cpu().numpy(), log_a.cpu().numpy()