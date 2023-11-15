import numpy as np
import math
import random

class QLearningAgent:
    def __init__(self, state_space_size, action_space_size, learning_rate, discount_factor, exploration_rate):
        self.q_table = np.zeros((state_space_size, action_space_size))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate

    def choose_action(self, state):
        if random.uniform(0, 1) < self.exploration_rate:
            return random.randint(0, self.q_table.shape[1] - 1)  # self.q_table.shape[1] = action_space_size
        else:
            return np.argmax(self.q_table[state, :])  # Exploit

    def update_q_table(self, state, action, reward, next_state):
        print(f"State index: {state}, Type: {type(state)}")
        print(f"Action: {action}, Type: {type(action)}")
        print(f"New State index: {next_state}, Type: {type(next_state)}")
        #Q(s,a) = Q(s,a) + learning_rate * (reward + discount_factor * max_future_q - Q(s,a))
        current_q = self.q_table[state, action]
        max_future_q = np.max(self.q_table[next_state, :]) 
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_future_q - current_q)
        self.q_table[state, action] = new_q
