# -*- coding: utf-8 -*-

import numpy as np
import random
import matplotlib.pyplot as plt

import copy
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from replay_buffer import ReplayBuffer


class QNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=100):
        super(QNetwork, self).__init__()
        self.lin1 = nn.Linear(input_size, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, output_size)

        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        x = self.lin1(x)
        x = self.bn1(x).relu()
        x = self.lin2(x)
        x = self.bn2(x).relu()
        res = self.lin3(x)
        return res # torch.softmax(self.lin3(res), dim=-1)


class DQNAgent:

    def __init__(
        self, num_hands=3, gamma=0.99, lr=0.001, batch_size=32, name='DQN', num_repetition=3,
        capacity=100, e=0.1, clipping_value=0.01, num_hold=3, hidden_size=200
    ):

        self.agent_type = 'dqn'
        self.name = name              # 名前
        self.num_hands = num_hands    # 手の数
        self.gamma = gamma            # 割引率
        self.batch_size = batch_size  # 学習時のバッチサイズ
        self.num_hold = num_hold      # 保持する直近の対戦履歴の数
        self.state = (([1] + [0] * (self.num_hands-1)) * 2) * self.num_hold  # 状態：list型
        self.qnet = QNetwork(input_size=self.num_hands*2*self.num_hold, output_size=self.num_hands, hidden_size=hidden_size)  # Q-Network
        self.target_qnet = copy.deepcopy(self.qnet)  # target Q-Net
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=lr)  # Q-Netのパラメータ更新に使うoptimizer
        self.memory = ReplayBuffer(capacity)
        self.e = e                            # epsilon for ε-greedy
        self.clipping_value = clipping_value  # 勾配クリッピング

    def update_q(self):
        """
        Q関数の更新
        ReplayBufferから対戦履歴を取得し，qnetのパラメータを更新する
        """
        batch = self.memory.sample(self.batch_size)     # 対戦履歴をReplayBufferからサンプリング
        q_infer_all = self.qnet(torch.from_numpy(batch['states']).float())  # qnetによるq値の算出
        # print(q_infer_all)
        action_mask = torch.tensor([list(batch['states'][i][-self.num_hands * 2:-self.num_hands]) for i in range(q_infer_all.shape[0])])
        q_infer = torch.sum(action_mask*q_infer_all, dim=1)
        # print(q_infer)

        # 教師信号の計算
        maxq = self.target_qnet(torch.tensor(batch['next_states'], dtype=torch.float)).max(1).values
        q_target = (torch.from_numpy(batch['rewards']) + self.gamma * maxq).float()

        self.qnet.train()
        self.optimizer.zero_grad()  # 勾配をリセット
        loss = nn.HuberLoss()(q_infer, q_target).float()  # 誤差：HuberLoss, MSELossなど

        loss.backward()  # バックプロパゲーションを計算
        torch.nn.utils.clip_grad_norm_(self.qnet.parameters(), self.clipping_value)  # 勾配クリッピング

        self.optimizer.step()  # 結合パラメータを更新
        self.target_qnet = copy.deepcopy(self.qnet)
        return loss

    def act(self):
        """行動（出す手の）決定：ε-greedy"""
        if self.e <= np.random.uniform(0, 1):
            self.qnet.eval()
            with torch.no_grad():
                q = self.qnet(torch.tensor(self.state, dtype=torch.float))
                action = q.argmax().item()
        else:
            action = random.choice(list(range(self.num_hands)))
        return action

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
    from agent_alpharandom import alphaRandomAgent

    """
    7手ジャンケン
    """
    rule7 = np.zeros((7, 7))
    edges = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [1, 6], [2, 3], [2, 5], [3, 4], [4, 0],
            [4, 1], [4, 2], [4, 5], [4, 6], [5, 0], [5, 1], [5, 3], [5, 6], [6, 0], [6, 2], [6, 3]]

    for e in edges:
        rule7[e[0]][e[1]] = 1
        rule7[e[1]][e[0]] = -1

    """
    DQN vs AlphaRandomで学習させる
    """
    CAPACITY = 10000
    agent1 = DQNAgent(
        name='DQN', num_hands=7, lr=0.0001, clipping_value=0.01,
        capacity=CAPACITY, num_hold=3, batch_size=64, gamma=0.99
    )
    agent2 = alphaRandomAgent(name='alphaRandom', num_hands=7, capacity=CAPACITY, num_hold=3, p=[0.073, 0.073, 0.027, 0.01, 0.544, 0.2, 0.073])
    game = RockPaperScissors(agent1, agent2, rule7)

    initial_memory_size = CAPACITY
    num_episodes = 1000
    max_steps = 100

    rewards_history = [[], []]

    agent1.e = 1.
    # agent2.e = 1.
    for i in range(initial_memory_size):
        agent1_ex_state, agent2_ex_state = agent1.state.copy(), agent2.state.copy()
        a1, a2, res = game.play()
        agent1.get_result([a1, a2])
        agent2.get_result([a2, a1])
        agent1.memorize(agent1_ex_state, a1, res, agent1.state)
        agent2.memorize(agent2_ex_state, a2, int(res*(-1)), agent2.state)


    for episode in range(num_episodes):
        game.reset_states()
        agent1.e = min(0.7, 0.995 ** episode)
        episode_reward = [0, 0]
        losses = []
        for step in range(max_steps):

            agent1_ex_state, agent2_ex_state = agent1.state.copy(), agent2.state.copy()
            a1, a2, res = game.play()

            loss1 = agent1.update_q()
            loss2 = torch.tensor(0., dtype=float)
            losses.append(loss1.item() + loss2.item())

            agent1.get_result([a1, a2])
            agent2.get_result([a2, a1])
            agent1.memorize(agent1_ex_state, a1, res, agent1.state)
            agent2.memorize(agent2_ex_state, a2, int(res*(-1)), agent2.state)
            episode_reward[0] += res
            episode_reward[1] += (res*(-1))

        rewards_history[0].append(episode_reward[0])
        rewards_history[1].append(episode_reward[1])
        if (episode+1) % 100 == 0:
            print('episode {} : agent1 {}, agent2 {}, loss {:.20f}'.format(
                episode+1, np.mean(rewards_history[0][-100:]), np.mean(rewards_history[1][-100:]), np.sum(losses)))


    plt.figure(figsize=(9, 6))
    x = np.arange(1, num_episodes)
    plt.plot(rewards_history[0], label='DQNAgent')
    plt.plot(rewards_history[1], label='alphaRandomAgent')
    plt.ylim(-105, 105)
    plt.xlabel('episode', fontsize='x-large')
    plt.ylabel('reward', fontsize='x-large')
    plt.xticks(fontsize='x-large')
    plt.yticks(fontsize='x-large')
    plt.legend(fontsize='x-large')

    save_path = '../figures/dqn.jpg'
    plt.savefig(save_path)
    plt.show()


