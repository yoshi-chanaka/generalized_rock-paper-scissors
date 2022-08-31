# -*- coding: utf-8 -*-

import random
import numpy as np
from replay_buffer import ReplayBuffer

class RandomAgent:

    def __init__(self, num_hands=3, name='Random', capacity=100, num_hold=3):

        self.agent_type = 'random'
        self.name = name                # 名前
        self.num_hands = num_hands  # 手の数
        self.num_hold = 3           # 保持する直近の対戦履歴の数
        self.state = (([1] + [0] * (self.num_hands-1)) * 2) * self.num_hold  # 状態：list型
        self.memory = ReplayBuffer(capacity)

    def act(self):
        """行動（出す手の）決定"""
        return random.choice(list(range(self.num_hands)))

    def get_result(self, result):
        """現在の状態を取得"""
        self.state = self.state[int(self.num_hands*2):] + np.identity(self.num_hands, dtype=int)[result[0]].tolist() + np.identity(self.num_hands, dtype=int)[result[1]].tolist()

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

    """
    3手ジャンケン
    """
    # 3手ジャンケンのルール（報酬）
    rule3 = np.array(
        [
            [0, 1, -1],
            [-1, 0, 1],
            [1, -1, 0]
        ]
    )
    agent1 = RandomAgent(name='RandomAgent1')
    agent2 = RandomAgent(name='RandomAgent2')
    game = RockPaperScissors(agent1, agent2, rule3)
    res = game.repeatedly_play(100000)
    print('3手ジャンケンの結果（10万回）: agent1 {}, agent2 {}, 引き分け {}\n'.format(res[0], res[1], res[2])) # 10万回勝負
    # 実行例：3手ジャンケンの結果（10万回）: agent1 33443, agent2 32914, 引き分け 33643

    # 1回ずつ結果を表示
    for i in range(20):
        game.play(False)

"""
draw: 1-1
draw: 2-2
RandomAgent1 win: 1-2
draw: 0-0
draw: 0-0
draw: 1-1
RandomAgent2 win: 2-1
draw: 1-1
RandomAgent2 win: 2-1
RandomAgent2 win: 0-2
RandomAgent2 win: 0-2
draw: 1-1
RandomAgent2 win: 0-2
RandomAgent1 win: 2-0
RandomAgent2 win: 0-2
RandomAgent2 win: 1-0
draw: 2-2
draw: 2-2
RandomAgent1 win: 2-0
RandomAgent2 win: 1-0
"""