# PPO (Proximal Policy Optimization) -> colab 실행

import torch
import torch.nn as nn
import torch.optim as optim

# PPO 정책 네트워크 정의
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.fc(x)

# PPO 업데이트 함수
def ppo_update(policy, optimizer, states, actions, advantages, old_probs, clip_epsilon=0.2):
    new_probs = policy(states).gather(1, actions)
    ratio = new_probs / (old_probs + 1e-6)
    clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
    
    loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 환경 설정
state_dim = 4
action_dim = 2
num_samples = 5  # 실행 가능하도록 샘플 개수 줄임

# PPO 테스트
ppo_policy = PolicyNetwork(state_dim, action_dim)
ppo_optimizer = optim.Adam(ppo_policy.parameters(), lr=0.001)

states = torch.rand(num_samples, state_dim)
actions = torch.randint(0, action_dim, (num_samples, 1))
advantages = torch.rand(num_samples, 1)
old_probs = torch.rand(num_samples, 1)

# PPO 업데이트 실행 전후 출력
ppo_before = ppo_policy(states).detach().numpy()
ppo_update(ppo_policy, ppo_optimizer, states, actions, advantages, old_probs)
ppo_after = ppo_policy(states).detach().numpy()

# 결과 출력
ppo_before, ppo_after
