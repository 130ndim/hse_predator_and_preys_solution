import numpy as np

import scipy as sp
import scipy.spatial


class Reward:
    _lock = None

    def __init__(self, n_preys, penalize_dist=0.05, catch_reward=100, out_of_box_reward=-100):
        self.n_preys = n_preys
        self.penalize_dist = penalize_dist
        self.catch_reward = catch_reward
        self.out_of_box_reward = out_of_box_reward
        self.reset()

    def reset(self):
        self._lock = np.ones(self.n_preys).astype(bool)

    def __call__(self, state):
        pred_xy = np.array([[obj['x_pos'], obj['y_pos']] for obj in state['predators']])
        prey_xy = np.array([[obj['x_pos'], obj['y_pos']] for obj in state['preys']])
        # obst_xy = np.array([[obj['x_pos'], obj['y_pos']] for obj in state['obstacles']])

        pred_r = np.array([obj['radius'] for obj in state['predators']])
        prey_r = np.array([obj['radius'] for obj in state['preys']])
        # obst_r = np.array([obj['radius'] for obj in state['obstacles']])

        prey_is_dead = ~np.array([obj['is_alive'] for obj in state['preys']])

        catch_reward = prey_is_dead * self._lock * self.catch_reward

        # pred2pred = sp.spatial.distance_matrix(pred_xy, pred_xy) - pred_r.reshape(-1, 1) - pred_r
        pred2prey = sp.spatial.distance_matrix(pred_xy, prey_xy) - pred_r.reshape(-1, 1) - prey_r
        # pred2obst = sp.spatial.distance_matrix(pred_xy, obst_xy) - pred_r.reshape(-1, 1) - obst_r
        # prey2prey = sp.spatial.distance_matrix(prey_xy, prey_xy) - prey_r.reshape(-1, 1) - prey_r
        # prey2obst = sp.spatial.distance_matrix(prey_xy, obst_xy) - prey_r.reshape(-1, 1) - obst_r
        pred_reward = -np.mean(pred2prey[:, self._lock], axis=1)

        preds_who_catch = pred2prey[:, prey_is_dead * self._lock]
        if preds_who_catch.size > 0:
            preds_who_catch = preds_who_catch.argmin(0)
            pred_reward[preds_who_catch] += self.catch_reward
        self._lock = ~prey_is_dead

        prey_reward = np.mean(pred2prey, axis=0) - catch_reward
        prey_reward *= self._lock

        return {'predator_rewards': pred_reward.tolist(), 'prey_rewards': prey_reward.tolist()}
        # print(prey_reward)

        # print(pred_reward)
        # print(prey_reward)
        # print(pred2pred)
        # print(pred2prey)
        # print(pred2obst)
        # print(prey2prey)
        # print(prey2obst)


if __name__ == '__main__':
    from predators_and_preys_env.env import PredatorsAndPreysEnv
    import numpy as np
    from examples.simple_chasing_agents.agents import ChasingPredatorAgent
    from examples.simple_chasing_agents.agents import FleeingPreyAgent
    import time

    reward = Reward(5)

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
        reward(state)
        step_count += 1

        # print(f"step {step_count}")

