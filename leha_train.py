from predators_and_preys_env.env import PredatorsAndPreysEnv, DEFAULT_CONFIG
from leha.agents import PredatorAgent, PreyAgent
from leha.rewards import pred_reward
from leha.features import generate_graph_prey, generate_graph_predator

from leha.td3 import TD3
import torch

from tqdm import tqdm
import numpy as np
from sklearn.metrics import mean_squared_error

TRANSITIONS = 1000000
EPS = 0.2

if __name__ == "__main__":

    env = PredatorsAndPreysEnv(render=False)
    predator_agent = PredatorAgent()
    prey_agent = PreyAgent()

    episodes_sampled = 0
    steps_sampled = 0

    state_dict = env.reset()
    rewards = []
    next_features = generate_graph_prey(state_dict)
    print(next_features)
    print(DEFAULT_CONFIG['game']['num_preds'])
    td3 = TD3(state_dim=next_features.x.size(1),
              action_dim=1,
              edge_dim=next_features.edge_attr.size(1),
              num_entities=DEFAULT_CONFIG['game']['num_preys'])

    for i in tqdm(range(TRANSITIONS)):
        steps = 0

        train_features = next_features

        # Epsilon-greedy policy
        prey_action = td3.act(train_features)
        prey_action = np.clip(prey_action + EPS * np.random.randn(*prey_action.shape), -1, +1)

        next_state_dict, _, done = env.step(predator_agent.act(state_dict), prey_action)
        # reward = pred_reward(state_dict)

        # mse
        default_action = prey_agent.act(state_dict)
        # print(default_action)
        # print(prey_action.shape)
        reward = -mean_squared_error(np.array(default_action).reshape(1, -1), prey_action)

        next_features = generate_graph_prey(next_state_dict)

        td3.update((train_features, prey_action, next_features, reward, done))

        rewards.append(reward)
        state_dict = next_state_dict if not done else env.reset()

        if (i + 1) % 5_000 == 0:
            print(f"Step: {i + 1}, Reward mean: {np.mean(rewards)}, Reward std: {np.std(rewards)}")
            td3.save()
            # ddpg.save()
            rewards = []
