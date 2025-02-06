import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

import matplotlib
matplotlib.use('TkAgg')  # GTK4 대신 TkAgg 사용
import matplotlib.pyplot as plt

# 환경 설정
env = gym.make("Pendulum-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = env.action_space.high[0]

# Actor 네트워크
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))  # -1 ~ 1 값으로 변환
        return x * self.max_action

# Critic 네트워크
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# 경험 재현 버퍼
class ReplayBuffer:
    def __init__(self, max_size=100000):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

# DDPG 에이전트
class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.critic = Critic(state_dim, action_dim)
        self.target_actor = Actor(state_dim, action_dim, max_action)
        self.target_critic = Critic(state_dim, action_dim)

        # 타겟 네트워크 초기화
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        # 최적화 설정
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-3)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.replay_buffer = ReplayBuffer()
        self.gamma = 0.99  # 할인율
        self.tau = 0.005   # Soft Update 계수

    def update(self, batch_size=64):
        if len(self.replay_buffer.buffer) < batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # Critic 학습 (TD Target)
        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            target_Q = self.target_critic(next_states, next_actions)
            target_Q = rewards + (self.gamma * target_Q * (1 - dones))

        current_Q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_Q, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor 학습
        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 타겟 네트워크 소프트 업데이트
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

# 학습 실행
agent = DDPGAgent(state_dim, action_dim, max_action)
num_episodes = 500
batch_size = 64

reward_list = []

for episode in range(num_episodes):
    state = env.reset()[0]  # 환경 초기화
    episode_reward = 0

    for step in range(200):  # 최대 200 스텝 진행
        noise = np.random.normal(0, 0.1, size=action_dim)  # 탐색 노이즈 추가
        action = agent.actor(torch.FloatTensor(state)).detach().numpy() + noise
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated  # 종료 여부 확인

        agent.replay_buffer.add(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward

        agent.update(batch_size)

    reward_list.append(episode_reward)
    print(f"Episode {episode}: Reward = {episode_reward:.2f}")

env.close()
plt.plot(reward_list)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("DDPG Training Progress")
plt.show()
