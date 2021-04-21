from abc import ABC

import numpy as np
import gym


class FoodTruck(gym.Env, ABC):

    def __init__(self):
        self.v_demand = [100, 200, 300, 400]
        self.p_demand = [.3, .4, .2, .1]
        self.capacity = self.v_demand[-1]
        self.days = ['Mon', 'Tue', 'Wen', 'Thu', 'Fri', 'Weekend']
        self.unit_cost, self.net_revenue = 4, 7
        self.action_space = [0, 100, 200, 300, 400]
        self.state_space = [('Mon', 0)] + [(d, i) for d in self.days[1:] for i in [0, 100, 200, 300]]

    def get_next_state_reward(self, state, action, demand):
        """
        :return: result dictionary with info
        """
        day, inventory = state
        result = {}
        result['next_day'] = self.days[self.days.index(day) + 1]
        result['staring_inventory'] = min(self.capacity, inventory + action)
        result['cost'] = self.unit_cost * action
        result['sales'] = min(result['staring_inventory'], demand)
        result['revenue'] = self.net_revenue * result['sales']
        result['next_inventory'] = result['staring_inventory'] - result['sales']
        result['reward'] = result['revenue'] - result['cost']
        return result

    def get_transition_prob(self, state, action):
        """
        :return: best_s_r_prob dictionary with all possible transitions and rewards for given state-action
        """
        next_s_r_prob = {}
        for demand in self.v_demand:
            result = self.get_next_state_reward(state, action, demand)
            next_s = result['next_day'], result['next_inventory']
            reward = result['reward']
            prob = self.p_demand[self.v_demand.index(demand)]
            if (next_s, reward) not in next_s_r_prob:
                next_s_r_prob[next_s, reward] = prob
            else:
                next_s_r_prob[next_s, reward] += prob
        return next_s_r_prob


a = FoodTruck()


















































































































































































































































































































































































































































