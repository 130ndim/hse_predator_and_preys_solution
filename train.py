import numpy as np

import torch

from tqdm.auto import tqdm

from agents.ddpg import DDPGAgent, DDPGConfig
from agents import CriticConfig, ActorConfig

from utils import Buffer
from predators_and_preys_env.env import PredatorsAndPreysEnv


def evaluate_policy(env, predator_agent, prey_agent, episodes=5):
    predator_agent.eval()
    prey_agent.eval()
    returns = []
    for i in range(episodes):
        env.seed(777 + i)
        done = False
        state = env.reset()
        pred_reward = prey_reward = 0.

        while not done:
            state, reward, done = env.step(predator_agent.act(state), prey_agent.act(state))
            pred_reward += np.sum(reward['predators'])
            prey_reward += np.sum(reward['preys'])
        returns.append((pred_reward, prey_reward))

    returns = np.array(returns)
    return returns.mean(0), returns.std(0)


if __name__ == '__main__':
    buffer_steps = 100000
    steps = 2000000 + buffer_steps
    batch_size = 64

    predator_config = DDPGConfig(critic=CriticConfig(input_size=(3, 2, 3), entity='predator'),
                                 actor=ActorConfig(entity='predator'),
                                 entity='predator')
    prey_config = DDPGConfig(critic=CriticConfig(input_size=(2, 3, 3), entity='prey'),
                             actor=ActorConfig(entity='prey'),
                             entity='prey')

    env = PredatorsAndPreysEnv(render=False)
    predator_agent = DDPGAgent(predator_config).cuda(2)
    prey_agent = DDPGAgent(prey_config).cuda(2)

    print(next(predator_agent.actor.parameters()).device)

    buffer = Buffer(buffer_size=500000)

    global_step_count = 0
    done = True
    step_count = 0
    state = env.reset()
    # buffer_bar = tqdm(total=buffer_steps)
    global_bar = tqdm(total=steps)

    pred_loss = []
    prey_loss = []
    rewards = []
    while True:
        predator_agent.train()
        prey_agent.train()
        if global_step_count < buffer_steps:
            predator_actions = np.random.rand(env.predator_action_size) * 2 - 1
            prey_actions = np.random.rand(env.prey_action_size) * 2 - 1
            # buffer_bar.update(1)
        else:
            # buffer_bar.disable = True
            predator_actions = predator_agent.act(state)
            prey_actions = prey_agent.act(state)

            loss = predator_agent.update(buffer.get_batch(batch_size, 'predator'))
            pred_loss.append(loss)
            # global_bar.set_postfix(loss)

            loss = prey_agent.update(buffer.get_batch(batch_size, 'prey'))
            prey_loss.append(loss)
            # global_bar.set_postfix(loss)

        next_state, reward, done = env.step(predator_actions, prey_actions)

        # print(next_state)
        buffer.append(
            state,
            {'predator_actions': predator_actions.tolist(), 'prey_actions': prey_actions.tolist()},
            next_state,
            reward,
            done
        )
        step_count += 1

        if done:
            state = env.reset()
            step_count = 0
        else:
            state = next_state

        global_step_count += 1

        global_bar.update(1)
        if global_step_count > buffer_steps:
            if (global_step_count) % 20000 == 0:
                mean, std = evaluate_policy(env, predator_agent, prey_agent)
                rewards.append((mean, std))
                global_bar.set_description(f'pred_r = {int(mean[0])}({int(std[0])}) | '
                                           f'prey_r = {int(mean[1])}({int(std[1])})')

            if (global_step_count) % 50000 == 0:
                predator_agent.save(f'/mnt/tank/scratch/dleonov/models/rl/ddpg_pred_{global_step_count}.pt')
                prey_agent.save(f'/mnt/tank/scratch/dleonov/models/rl/ddpg_prey_{global_step_count}.pt')

            if (global_step_count) % 200000 == 0:
                torch.save(pred_loss, '/mnt/tank/scratch/dleonov/pred_loss.pt')
                torch.save(prey_loss, '/mnt/tank/scratch/dleonov/prey_loss.pt')
                torch.save(rewards, '/mnt/tank/scratch/dleonov/rewards.pt')

