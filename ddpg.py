# individual network settings for each actor + critic pair
# see networkforall for details

from networkforall import Network, Actor, Critic
from utilities import hard_update, gumbel_softmax, onehot_from_logits
from torch.optim import Adam
import torch
import numpy as np


# add OU noise for exploration
from OUNoise import OUNoise

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = 'cpu'

class DDPGAgent:
    def __init__(self, is_unity, in_actor, hidden_in_actor, hidden_out_actor, out_actor, in_critic, hidden_in_critic, hidden_out_critic, lr_actor=1.0e-3, lr_critic=1.0e-3):
        super(DDPGAgent, self).__init__()

        if is_unity is False:
            self.actor = Network(in_actor, hidden_in_actor, hidden_out_actor, out_actor, actor=True).to(device)
            self.critic = Network(in_critic, hidden_in_critic, hidden_out_critic, 1).to(device)
            self.target_actor = Network(in_actor, hidden_in_actor, hidden_out_actor, out_actor, actor=True).to(device)
            self.target_critic = Network(in_critic, hidden_in_critic, hidden_out_critic, 1).to(device)
        else:
            self.seed = 0
            self.actor = Actor(in_actor, out_actor, self.seed).to(device)
            self.critic = Critic(in_critic, self.seed).to(device)
            self.target_actor = Actor(in_actor, out_actor, self.seed).to(device)
            self.target_critic = Critic(in_critic, self.seed).to(device)
        self.noise = OUNoise(out_actor, scale=1.0)

        
        # initialize targets same as original networks
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr_critic)


    def act(self, obs, noise=0.0):
        obs = obs.to(device)
        action = self.actor(obs) + noise*self.noise.noise()
        return torch.from_numpy(np.clip(action.detach().numpy(), -1, 1))

    def target_act(self, obs, noise=0.0):
        obs = obs.to(device)
        action = self.target_actor(obs) + noise*self.noise.noise()
        return action.clamp(-1, 1)
