# -*- coding: utf-8 -*-

import numpy as np

class RockPaperScissors:

    def __init__(self, agent1, agent2, rule):

        self.agent1 = agent1
        self.agent2 = agent2
        self.rule = rule  # square matrix, ndarray

        if rule.shape[0] < np.min([self.agent1.num_hands, self.agent2.num_hands]) or \
                agent1.num_hands != agent2.num_hands:
            print("agents cannot play RPS")

    def play(self, return_data=True):
        """1回ジャンケン"""
        a1, a2 = self.agent1.act(), self.agent2.act()
        res = self.rule[a1][a2]

        if return_data:
            return a1, a2, res
        else:
            if res == 1:
                print('{} win: {}-{}'.format(self.agent1.name, a1, a2))
            elif res == -1:
                print('{} win: {}-{}'.format(self.agent2.name, a1, a2))
            else:
                print('draw: {}-{}'.format(a1, a2))

    def repeatedly_play(self, num_iters=10):
        """繰り返しジャンケン"""
        cnt = [0, 0, 0]  # agent1の勝ち回数, agent2の勝ち回数，引き分け回数
        for i in range(num_iters):
            a1, a2 = self.agent1.act(), self.agent2.act()
            res = self.rule[a1][a2]
            if res == 1:
                cnt[0] += 1
            elif res == -1:
                cnt[1] += 1
            else:
                cnt[2] += 1
        return cnt

    def reset_states(self):
        """各エージェントの現在の状態をランダムに設定"""
        hands_idx1 = np.random.choice(np.arange(0, 7).tolist(), size=int(max(self.agent1.num_hold, self.agent2.num_hold)))
        hands_idx2 = np.random.choice(np.arange(0, 7).tolist(), size=int(max(self.agent1.num_hold, self.agent2.num_hold)))
        self.agent1.state, self.agent2.state = [], []
        for i in range(int(self.agent1.num_hold)):
            self.agent1.state += np.identity(self.agent1.num_hands, dtype=int)[hands_idx1[i]].tolist()
            self.agent1.state += np.identity(self.agent1.num_hands, dtype=int)[hands_idx2[i]].tolist()
        for i in range(int(self.agent2.num_hold)):
            self.agent2.state += np.identity(self.agent1.num_hands, dtype=int)[hands_idx2[i]].tolist()
            self.agent2.state += np.identity(self.agent1.num_hands, dtype=int)[hands_idx1[i]].tolist()


if __name__ == "__main__":
    """
    ジャンケンのルールの設定
    """
    # 3手ジャンケンのルール（報酬）
    rule3 = np.array(
        [
            [0, 1, -1],
            [-1, 0, 1],
            [1, -1, 0]
        ]
    )

    # 7手ジャンケン
    rule7 = np.zeros((7, 7))
    edges = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [1, 6], [2, 3], [2, 5], [3, 4], [4, 0],
            [4, 1], [4, 2], [4, 5], [4, 6], [5, 0], [5, 1], [5, 3], [5, 6], [6, 0], [6, 2], [6, 3]]

    for e in edges:
        rule7[e[0]][e[1]] = 1
        rule7[e[1]][e[0]] = -1
    print(rule7)

"""
[[ 0.  1.  1.  1. -1. -1. -1.]
 [-1.  0.  1.  1. -1. -1.  1.]
 [-1. -1.  0.  1. -1.  1. -1.]
 [-1. -1. -1.  0.  1. -1. -1.]
 [ 1.  1.  1. -1.  0.  1.  1.]
 [ 1.  1. -1.  1. -1.  0.  1.]
 [ 1. -1.  1.  1. -1. -1.  0.]]
 """