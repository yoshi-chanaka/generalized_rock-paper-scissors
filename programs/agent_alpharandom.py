# -*- coding: utf-8 -*-

import numpy as np
from replay_buffer import ReplayBuffer


class alphaRandomAgent:

    def __init__(self, num_hands=3, name='Random', capacity=100, num_hold=3, p=None):

        self.agent_type = 'alpha_random'
        self.name = name            # 名前
        self.num_hands = num_hands  # 手の数
        self.num_hold = 3           # 保持する直近の対戦履歴の数
        self.state = (([1] + [0] * (self.num_hands-1)) * 2) * self.num_hold  # 状態：list型
        self.memory = ReplayBuffer(capacity)

        # 出す手の確率分布
        if p == None:
            self.p = [1/self.num_hands] * self.num_hands
        else:
            self.p = p

    def act(self):
        """行動（出す手の）決定"""
        return np.random.choice(list(range(self.num_hands)), p=self.p)

    def get_result(self, result):
        """現在の状態を取得"""
        self.state = \
            self.state[int(self.num_hands*2):] + \
            np.identity(self.num_hands, dtype=int)[result[0]].tolist() + \
            np.identity(self.num_hands, dtype=int)[result[1]].tolist()

    def memorize(self, state, action, reward, state_next):
        """s, a, r, s' を記録"""
        self.memory.memorize(state, action, reward, state_next)

    def reset_state(self):
        """現在の状態をランダムに設定"""
        hands_idx = np.random.choice(np.arange(0, 7).tolist(), size=int(self.num_hold * 2))
        self.state = []
        for hand in hands_idx:
            self.state += np.identity(self.num_hands, dtype=int)[hand].tolist()


if __name__ == "__main__":
    from game import RockPaperScissors
    from agent_random import RandomAgent

    """
    7手ジャンケン
    """
    rule7 = np.zeros((7, 7))
    edges = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [1, 6], [2, 3], [2, 5], [3, 4], [4, 0],
            [4, 1], [4, 2], [4, 5], [4, 6], [5, 0], [5, 1], [5, 3], [5, 6], [6, 0], [6, 2], [6, 3]]

    for e in edges:
        rule7[e[0]][e[1]] = 1
        rule7[e[1]][e[0]] = -1

    agent1 = RandomAgent(name='RandomAgent', num_hands=7)
    agent2 = alphaRandomAgent(name='alphaRandomAgent', num_hands=7, num_hold=7, p=[0.073, 0.073, 0.027, 0.01, 0.544, 0.2, 0.073])
    game = RockPaperScissors(agent1, agent2, rule7)
    res = game.repeatedly_play(100000)
    print('7手ジャンケンの結果（10万回）: {} {}, {} {}, 引き分け {}\n'.format(agent1.name, res[0], agent2.name, res[1], res[2]))  # 10万回勝負
    # 実行例：7手ジャンケンの結果（10万回）: RandomAgent 25300, alphaRandomAgent 60464, 引き分け 14236

    # 1回ずつ結果を表示
    for i in range(20):
        game.play(False)

"""
alphaRandomAgent win: 6-1
alphaRandomAgent win: 3-6
alphaRandomAgent win: 6-4
alphaRandomAgent win: 5-4
alphaRandomAgent win: 2-0
alphaRandomAgent win: 0-4
alphaRandomAgent win: 6-5
alphaRandomAgent win: 0-4
alphaRandomAgent win: 6-4
draw: 0-0
RandomAgent win: 3-4
RandomAgent win: 3-4
alphaRandomAgent win: 5-4
alphaRandomAgent win: 1-4
RandomAgent win: 2-5
alphaRandomAgent win: 5-4
alphaRandomAgent win: 0-4
RandomAgent win: 1-6
alphaRandomAgent win: 2-0
alphaRandomAgent win: 5-4
"""
