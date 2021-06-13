import numpy as np

import torch
from torch_geometric.data import Data


def generate_graph_predator(state_dict):
    features = []

    edge_features = []
    edge_index = []

    for i, predator in enumerate(state_dict['predators']):
        x_pred, y_pred, r_pred, speed_pred = predator['x_pos'], predator['y_pos'], predator['radius'], predator['speed']

        features.append([x_pred, y_pred, r_pred, 1, 0, 0])

        for i_, predator in enumerate(state_dict['predators']):
            if i == i_:
                continue
            x_pred_, y_pred_ = predator['x_pos'], predator['y_pos']

            angle = np.arctan2(y_pred_ - y_pred, x_pred_ - x_pred) / np.pi
            distance = np.sqrt((y_pred_ - y_pred) ** 2 + (x_pred_ - x_pred) ** 2)

            edge_features.append([angle, distance, 1, 0, 0])
            edge_index.append([i_, i])

        for j, prey in enumerate(state_dict['preys'], start=len(state_dict['predators'])):
            x_prey, y_prey, r_prey, speed_prey, alive = prey['x_pos'], prey['y_pos'], \
                                                        prey['radius'], prey['speed'], prey['is_alive']
            if not alive:
                continue

            angle = np.arctan2(y_prey - y_pred, x_prey - x_pred) / np.pi
            distance = np.sqrt((y_prey - y_pred) ** 2 + (x_prey - x_pred) ** 2)

            edge_features.append([angle, distance, 0, 1, 0])
            edge_index.append([j, i])

        for k, obs in enumerate(state_dict['obstacles'], start=len(state_dict['predators']) + len(state_dict['preys'])):
            x_obs, y_obs, r_obs = obs['x_pos'], obs['y_pos'], obs['radius']
            angle = np.arctan2(y_obs - y_pred, x_obs - x_pred) / np.pi
            distance = np.sqrt((y_obs - y_pred) ** 2 + (x_obs - x_pred) ** 2)

            edge_features.append([angle, distance, 0, 0, 1])
            edge_index.append([k, i])

    for prey in state_dict['preys']:
        x_prey, y_prey, r_prey, speed_prey, alive = prey['x_pos'], prey['y_pos'], \
                                                    prey['radius'], prey['speed'], prey['is_alive']
        if not alive:
            continue

        features.append([x_prey, y_prey, r_prey, 0, 1, 0])

    for obs in state_dict['obstacles']:
        x_obs, y_obs, r_obs = obs['x_pos'], obs['y_pos'], obs['radius']
        features.append([x_obs, y_obs, r_obs, 0, 0, 1])

    len_preds = len(state_dict['predators'])

    return Data(
        x=torch.tensor(features, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long).T,
        edge_attr=torch.tensor(edge_features, dtype=torch.float),
        mask=torch.tensor([1] * len_preds + [0] * (len(features) - len_preds), dtype=torch.bool)
    )


def generate_graph_prey(state_dict):
    features = []

    edge_features = []
    edge_index = []

    for i, prey in enumerate(state_dict['preys']):
        x_prey, y_prey, r_prey, speed_prey, alive = prey['x_pos'], prey['y_pos'], prey['radius'], prey['speed'], prey['is_alive']

        features.append([x_prey, y_prey, r_prey, 1, 0, 0] + ([0, 0, 1] if alive else [0, 1, 0]))

        for i_, prey in enumerate(state_dict['preys']):
            if i == i_:
                continue
            x_prey_, y_prey_ = prey['x_pos'], prey['y_pos']

            angle = np.arctan2(y_prey_ - y_prey, x_prey_ - x_prey) / np.pi
            distance = np.sqrt((y_prey_ - y_prey) ** 2 + (x_prey_ - x_prey) ** 2)

            edge_features.append([angle, distance, 1, 0, 0])
            edge_index.append([i_, i])

        for j, predator in enumerate(state_dict['predators'], start=len(state_dict['preys'])):
            x_pred, y_pred, r_pred, speed_pred = predator['x_pos'], predator['y_pos'], \
                                                        predator['radius'], predator['speed']

            angle = np.arctan2(y_prey - y_pred, x_prey - x_pred) / np.pi
            distance = np.sqrt((y_prey - y_pred) ** 2 + (x_prey - x_pred) ** 2)

            edge_features.append([angle, distance, 0, 1, 0])
            edge_index.append([j, i])

        for k, obs in enumerate(state_dict['obstacles'], start=len(state_dict['predators']) + len(state_dict['preys'])):
            x_obs, y_obs, r_obs = obs['x_pos'], obs['y_pos'], obs['radius']
            angle = np.arctan2(y_obs - y_prey, x_obs - x_prey) / np.pi
            distance = np.sqrt((y_obs - y_prey) ** 2 + (x_obs - x_prey) ** 2)

            edge_features.append([angle, distance, 0, 0, 1])
            edge_index.append([k, i])

    for predator in state_dict['predators']:
        x_prey, y_prey, r_prey, speed_prey = predator['x_pos'], predator['y_pos'], \
                                                    predator['radius'], predator['speed']

        features.append([x_prey, y_prey, r_prey, 0, 1, 0, 1, 0, 0])

    for obs in state_dict['obstacles']:
        x_obs, y_obs, r_obs = obs['x_pos'], obs['y_pos'], obs['radius']
        features.append([x_obs, y_obs, r_obs, 0, 0, 1, 1, 0, 0])

    len_preys = len(state_dict['preys'])

    return Data(
        x=torch.tensor(features, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long).T,
        edge_attr=torch.tensor(edge_features, dtype=torch.float),
        mask=torch.tensor([1] * len_preys + [0] * (len(features) - len_preys), dtype=torch.bool)
    )


def generate_features_predator(state_dict):
    features = []

    for predator in state_dict['predators']:
        x_pred, y_pred, r_pred, speed_pred = predator['x_pos'], predator['y_pos'], predator['radius'], predator['speed']

        features += [x_pred, y_pred]

        prey_list = []

        for prey in state_dict['preys']:
            x_prey, y_prey, r_prey, speed_prey, alive = prey['x_pos'], prey['y_pos'], \
                                                        prey['radius'], prey['speed'], prey['is_alive']
            angle = np.arctan2(y_prey - y_pred, x_prey - x_pred) / np.pi
            distance = np.sqrt((y_prey - y_pred) ** 2 + (x_prey - x_pred) ** 2)

            prey_list += [[angle, distance, int(alive), r_prey]]

        prey_list = sorted(prey_list, key=lambda x: x[1])
        prey_list = [item for sublist in prey_list for item in sublist]
        features += prey_list

        obs_list = []

        for obs in state_dict['obstacles']:
            x_obs, y_obs, r_obs = obs['x_pos'], obs['y_pos'], obs['radius']
            angle = np.arctan2(y_obs - y_pred, x_obs - x_pred) / np.pi
            distance = np.sqrt((y_obs - y_pred) ** 2 + (x_obs - x_pred) ** 2)

            obs_list += [[angle, distance, r_obs]]

        obs_list = sorted(obs_list, key=lambda x: x[1])
        obs_list = [item for sublist in obs_list for item in sublist]
        features += obs_list

    return np.array(features, dtype=np.float32)


def generate_features_prey(state_dict):
    features = []

    for prey in state_dict['preys']:
        x_prey, y_prey, r_prey, speed_prey, alive = prey['x_pos'], prey['y_pos'], \
                                                    prey['radius'], prey['speed'], prey['is_alive']

        features += [x_prey, y_prey, alive, r_prey]

        pred_list = []

        for predator in state_dict['predators']:
            x_pred, y_pred, r_pred, speed_pred = predator['x_pos'], predator['y_pos'], predator['radius'], predator[
                'speed']

            angle = np.arctan2(y_prey - y_pred, x_prey - x_pred) / np.pi
            distance = np.sqrt((y_prey - y_pred) ** 2 + (x_prey - x_pred) ** 2)

            pred_list += [[angle, distance, int(alive), r_prey]]

        pred_list = sorted(pred_list, key=lambda x: x[1])
        pred_list = [item for sublist in pred_list for item in sublist]
        features += pred_list

        obs_list = []

        for obs in state_dict['obstacles']:
            x_obs, y_obs, r_obs = obs['x_pos'], obs['y_pos'], obs['radius']
            angle = np.arctan2(y_obs - y_prey, x_obs - x_prey) / np.pi
            distance = np.sqrt((y_obs - y_prey) ** 2 + (x_obs - x_prey) ** 2)

            obs_list += [[angle, distance, r_obs]]

        obs_list = sorted(obs_list, key=lambda x: x[1])
        obs_list = [item for sublist in obs_list for item in sublist]
        features += obs_list

    return np.array(features, dtype=np.float32)


if __name__ == '__main__':
    from predators_and_preys_env.env import PredatorsAndPreysEnv
    from torch_geometric.nn import NNConv
    from torch.nn import Linear
    env = PredatorsAndPreysEnv(render=False)
    state = env.reset()
    pred_graph = generate_graph_predator(state)
    print(pred_graph.x)
    nn = Linear(5, 36, bias=False)
    net = NNConv(6, 6, nn=nn, bias=False, root_weight=False)
    net.nn.weight.data.fill_(0.001)

    print(net(pred_graph.x, pred_graph.edge_index, pred_graph.edge_attr))

    # print(generate_graph_prey(state).edge_index)
