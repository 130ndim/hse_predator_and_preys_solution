import numpy as np

import scipy as sp
import scipy.spatial


class Penalty:
    # _lock = None

    def __init__(self, penalty=1., max_dist=9., penalize_dist=0.075, penalize_preys=True):
        self.penalty = penalty
        self.max_dist = max_dist
        self.penalize_dist = penalize_dist
        self.penalize_preys = penalize_preys

    def __call__(self, state, reward):
        pred_xy = np.array([[obj['x_pos'], obj['y_pos']] for obj in state['predators']])
        obst_xy = np.array([[obj['x_pos'], obj['y_pos']] for obj in state['obstacles']])

        pred_r = np.array([obj['radius'] for obj in state['predators']])
        obst_r = np.array([obj['radius'] for obj in state['obstacles']])

        # prey_is_dead = ~np.array([obj['is_alive'] for obj in state['preys']])

        # pred2pred = sp.spatial.distance_matrix(pred_xy, pred_xy) - pred_r.reshape(-1, 1) - pred_r
        # pred2prey = sp.spatial.distance_matrix(pred_xy, prey_xy) - pred_r.reshape(-1, 1) - prey_r
        pred2obst = sp.spatial.distance_matrix(pred_xy, obst_xy) - pred_r.reshape(-1, 1) - obst_r
        obst_penalty = np.isclose(pred2obst, 0, atol=self.penalize_dist, rtol=0).sum(1) * -self.penalty
        wall_penalty = np.isclose(self.max_dist - np.abs(pred_xy) - pred_r.reshape(-1, 1), 0, atol=self.penalize_dist, rtol=0).sum(1) * -self.penalty

        reward['predators'] += obst_penalty
        reward['predators'] += wall_penalty

        if self.penalize_preys:
            prey_xy = np.array([[obj['x_pos'], obj['y_pos']] for obj in state['preys']])
            prey_r = np.array([obj['radius'] for obj in state['preys']])

            prey2obst = sp.spatial.distance_matrix(prey_xy, obst_xy) - prey_r.reshape(-1, 1) - obst_r
            obst_penalty = np.isclose(prey2obst, 0, atol=self.penalize_dist, rtol=0).sum(1) * -self.penalty
            wall_penalty = np.isclose(self.max_dist - np.abs(prey_xy) - prey_r.reshape(-1, 1), 0, atol=self.penalize_dist, rtol=0).sum(
                1) * -self.penalty

            reward['preys'] += obst_penalty
            reward['preys'] += wall_penalty

        return reward


if __name__ == '__main__':
    from predators_and_preys_env.env import PredatorsAndPreysEnv
    import numpy as np
    from examples.simple_chasing_agents.agents import ChasingPredatorAgent
    from examples.simple_chasing_agents.agents import FleeingPreyAgent
    import time

    reward = Penalty(5)

    env = PredatorsAndPreysEnv(render=True)
    predator_agent = ChasingPredatorAgent()
    prey_agent = FleeingPreyAgent()

    done = True
    step_count = 0
    state = None
    while True:
        if done:
            print("reset")
            state = env.reset()
            step_count = 0

        state, done = env.step(predator_agent.act(state), prey_agent.act(state))
        # print(state)
        reward(state, None)
        step_count += 1

        # print(f"step {step_count}")

