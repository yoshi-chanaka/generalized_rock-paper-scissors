# -*- coding: utf-8 -*-

import random
import numpy as np
from collections import namedtuple

class ReplayBuffer:

    def __init__(self, capacity=10000):
        self.capacity = capacity  # 容量
        self.memory = []         # メモリ
        self.index = 0
        self.Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))

    def sample(self, batch_size):
        """memoryからサンプリング"""
        samples_indices = random.sample(np.arange(0, len(self.memory)).tolist(), batch_size)
        samples_state = np.array([self.memory[i].state for i in samples_indices])
        samples_action = np.array([self.memory[i].action for i in samples_indices])
        samples_reward = np.array([self.memory[i].reward for i in samples_indices])
        samples_next = np.array([self.memory[i].next_state for i in samples_indices])
        return {'states': samples_state, 'actions': samples_action, 'rewards': samples_reward, 'next_states': samples_next}

    def memorize(self, state, action, reward, state_next):
        """memoryへの記録"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)  # メモリが満タンでないときは足す

        self.memory[self.index] = self.Transition(state, action, reward, state_next)
        self.index = (self.index + 1) % self.capacity  # 保存するindexを1つずらす


if __name__ == "__main__":
    pass