import numpy as np
#from predators_and_preys_env.agent import PredatorAgent, PreyAgent

def distance(first, second):
    return ((first["x_pos"] - second["x_pos"]) ** 2 + (first["y_pos"] - second["y_pos"]) ** 2) ** 0.5

EPS = 0.01
DELTA = 10
BACK_DELTA = 10

class PredatorAgent:
    def __init__(self):
        self.priv_state = None
        self.random_delay = [-1] * 3
        self.random_angle = [[-1] * BACK_DELTA + [-1] * DELTA] * 3

    def act(self, state_dict):
        action = []

        for j, predator in enumerate(state_dict["predators"]):
            closest_prey = None
            angle_value = 0.

            if self.random_delay[j] < 0:
                for i, prey in enumerate(state_dict["preys"]):
                    if not prey["is_alive"]:
                        continue
                    if closest_prey is None:
                        closest_prey = prey
                    else:
                        if distance(closest_prey, predator) > distance(prey, predator):
                            closest_prey = prey

                if closest_prey is not None:
                    angle_value = np.arctan2(closest_prey["y_pos"] - predator["y_pos"],
                                             closest_prey["x_pos"] - predator["x_pos"]) / np.pi

                if self.priv_state is not None:
                    priv_predator = self.priv_state["predators"][j]

                    if abs(priv_predator["y_pos"] - predator["y_pos"]) + abs(
                            priv_predator["x_pos"] - predator["x_pos"]) < EPS:
                        self.random_delay[j] = DELTA + BACK_DELTA
                        self.random_angle[j] = [1 + angle_value] * BACK_DELTA + [np.random.uniform(-1, 1)] * DELTA
            else:
                self.random_delay[j] -= 1
                angle_value = self.random_angle[j][self.random_delay[j]]

            action.append(angle_value)

        self.priv_state = state_dict
        return action


class PreyAgent:
    def act(self, state_dict):
        action = []
        for prey in state_dict["preys"]:
            closest_predator = None
            for predator in state_dict["predators"]:
                if closest_predator is None:
                    closest_predator = predator
                else:
                    if distance(closest_predator, prey) > distance(prey, predator):
                        closest_predator = predator
            if closest_predator is None:
                action.append(0.)
            else:
                action.append(1 + np.arctan2(closest_predator["y_pos"] - prey["y_pos"],
                                             closest_predator["x_pos"] - prey["x_pos"]) / np.pi)
        return action