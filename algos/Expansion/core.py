import numpy as np
import scipy.signal
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import torch.nn.functional as F

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.

        pi = self._distribution(obs)

        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act).mean()
        return pi, logp_a


class MLPCategoricalActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.act_dim = act_dim

        state_dim = 3
        obs_dim = obs_dim * state_dim

        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):

        state_cost, state_flag, state_tonggou = obs
        obs = torch.cat((F.log_softmax(state_cost.reshape((len(state_cost), 1)), dim=0),
                         state_flag.reshape((len(state_cost), 1)), state_tonggou.reshape((len(state_cost), 1))),
                        axis=1).flatten()
        logits = self.logits_net(obs)

        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        state_dim = 3
        obs_dim = obs_dim * state_dim
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        state_cost, state_flag, state_tonggou = obs
        obs = torch.cat((F.log_softmax(state_cost.reshape((len(state_cost), 1)), dim=0),
                         state_flag.reshape((len(state_cost), 1)), state_tonggou.reshape((len(state_cost), 1))),
                        axis=1).flatten()
        return torch.squeeze(self.v_net(obs), -1)  # Critical to ensure v has right shape.


class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, adjust_num, cuda,
                 hidden_sizes=(64, 64), activation=nn.Tanh):
        super().__init__()
        obs_dim = observation_space.shape[0]
        # policy builder depends on action space
        self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)

        # build value function
        self.v = MLPCritic(obs_dim, hidden_sizes, activation)
        self.adjust_num = adjust_num
        if cuda:
            self.pi.cuda()
            self.v.cuda()

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample((self.adjust_num,))
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.flatten().cpu().numpy(), v.cpu().numpy(), logp_a.cpu().mean().numpy()

    def act(self, obs):
        return self.step(obs)[0]