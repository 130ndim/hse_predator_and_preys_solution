from copy import deepcopy
from dataclasses import dataclass

from typing import Optional
from typing_extensions import Literal

import numpy as np

import torch
from torch.nn import functional as F, utils
from torch.optim import Adam

from . import Agent
from .pyg import ActorConfig, PredatorActor, PreyActor, CriticConfig, PredatorCritic, \
    PreyCritic
from .utils import soft_update, ZeroCenteredNoise, OUNoise
from utils.pyg_buffer import PyGBatch, state2tensor


@dataclass
class DDPGConfig:
    actor: ActorConfig = ActorConfig()
    critic: CriticConfig = CriticConfig()

    noise: Literal['normal', 'ou'] = 'normal'
    add_noise_on_inference: bool = False

    gamma: float = 0.99
    tau: float = 0.995

    policy_update_freq: int = 12

    bar: bool = True

    entity: Optional[Literal['prey', 'predator']] = None
    n_entities: int = 1


class DDPGAgent(Agent):
    _bar = None

    def __init__(self, config: DDPGConfig = DDPGConfig()):
        self.config = config
        _actor_class = PredatorActor if config.entity == 'predator' else PreyActor
        _critic_class = PredatorCritic if config.entity == 'predator' else PreyCritic

        self.actor = _actor_class(config.actor)
        self.target_actor = deepcopy(self.actor)

        self.critic = _critic_class(config.critic)
        self.target_critic = deepcopy(self.critic)

        print('Actor:\n', self.actor)
        print('Critic:\n', self.critic)

        self.actor_optim = Adam(self.actor.parameters(), lr=config.actor.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=config.critic.lr)

        self.noise = OUNoise() if config.noise == 'ou' else ZeroCenteredNoise()

        self.add_noise_on_inference = config.add_noise_on_inference

        self._entity = config.entity
        self._step = 0

        self._training = True

    @torch.no_grad()
    def act(self, state):
        state = state2tensor(state).to(self.device)
        # print(state.edge_index)
        action = self.actor(state).squeeze().cpu().numpy()
        if self._training or self.add_noise_on_inference:
            action = self.noise(action)
        return np.clip(action, -1., 1.)

    def train(self):
        self._training = True

    def eval(self):
        self._training = False

    def update(self, batch: PyGBatch):
        self._step += 1

        state, action, next_state, reward, done = batch.to(self.device)
        batch_, mask = batch.state.batch, batch.state.mask
        B = batch_.max() + 1

        with torch.no_grad():
            next_action = self.target_actor(next_state)
            Q_target = self.target_critic(next_state, next_action).view(B, -1)
            Q_target = reward + self.config.gamma * Q_target * (1 - done)

        Q_est = self.critic(state, action).view(B, -1)

        self.critic_optim.zero_grad()
        value_loss = F.mse_loss(Q_est, Q_target)
        value_loss.backward()
        self.critic_optim.step()
        losses = {'v_loss': value_loss.item(), 'p_loss': np.nan}

        if self._step % self.config.policy_update_freq == 0:
            self.actor_optim.zero_grad()
            policy_loss = -torch.mean(self.critic(state, self.actor(state)))
            policy_loss.backward()
            utils.clip_grad_norm_(self.actor.parameters(), self.config.actor.max_grad_norm)
            self.actor_optim.step()
            losses['p_loss'] = policy_loss.item()

            soft_update(self.target_actor, self.actor, self.config.tau)
            soft_update(self.target_critic, self.critic, self.config.tau)

        return losses

    def save(self, path):
        state_dict = {'step': self._step,
                      'config': self.config,
                      'actor': self.actor.state_dict(),
                      'target_actor': self.target_actor.state_dict(),
                      'critic': self.critic.state_dict(),
                      'target_critic': self.target_critic.state_dict(),
                      'actor_optim': self.actor_optim.state_dict(),
                      'critic_optim': self.critic_optim.state_dict()}
        torch.save(state_dict, path)

    @classmethod
    def from_ckpt(cls, ckpt_path, map_location=None):
        state_dict = torch.load(ckpt_path, map_location=map_location)
        step, config = state_dict.pop('step'), state_dict.pop('config')

        agent = cls(config)
        agent._step = step
        for k, v in state_dict.items():
            getattr(agent, k).load_state_dict(v)

        return agent


if __name__ == '__main__':
    agent = DDPGAgent()
    agent.save('test.pt')
    print(agent.actor.net.seq[0].weight.data)
    agent2 = DDPGAgent.from_ckpt('test.pt')
    agent3 = agent2.from_ckpt('test.pt')
    print(agent2.actor.net.seq[0].weight.data)
    print(agent3.actor.net.seq[0].weight.data)
