# GRPO (Group Relative Policy Optimization) -> colab 실행

import torch
import torch.nn as nn
import torch.optim as optim

# GRPO 정책 네트워크 정의
class GRPOPolicy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(GRPOPolicy, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.fc(x)

# GRPO 상대적 보상 계산 함수
def compute_relative_rewards(group_rewards):
    mean_reward = torch.mean(group_rewards)
    std_reward = torch.std(group_rewards) + 1e-6  # 0으로 나누는 문제 방지
    relative_rewards = (group_rewards - mean_reward) / std_reward
    return relative_rewards

# GRPO 업데이트 함수
def grpo_update(policy, optimizer, states, actions, group_rewards):
    relative_rewards = compute_relative_rewards(group_rewards)
    new_probs = policy(states).gather(1, actions)
    loss = -(new_probs * relative_rewards).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 환경 설정
state_dim = 4
action_dim = 2
num_samples = 5  # 실행 가능하도록 샘플 개수 줄임

# GRPO 테스트
grpo_policy = GRPOPolicy(state_dim, action_dim)
grpo_optimizer = optim.Adam(grpo_policy.parameters(), lr=0.001)

states = torch.rand(num_samples, state_dim)
actions = torch.randint(0, action_dim, (num_samples, 1))
group_rewards = torch.rand(num_samples, 1) * 10  # 그룹 내 서로 다른 보상 값

# GRPO 업데이트 실행 전후 출력
grpo_before = grpo_policy(states).detach().numpy()
grpo_update(grpo_policy, grpo_optimizer, states, actions, group_rewards)
grpo_after = grpo_policy(states).detach().numpy()

# 결과 출력
print("=== GRPO 업데이트 실행 전 ===")
print(grpo_before)

print("\n=== GRPO 업데이트 실행 후 ===")
print(grpo_after)
