# 상태 가치 함수 V(s) 계산
gamma = 0.9  # 감가율
rewards = [1, 1, 1, 1, 1]  # 단순히 매 시점 보상이 1인 경우

# 첫 번째 식: 무한 시점 누적 보상 계산
def calculate_value_directly(rewards, gamma):
    V = 0
    for t, reward in enumerate(rewards):
        V += (gamma ** t) * reward
    return V

# 두 번째 식: 벨만 기대 방정식으로 계산
def calculate_value_recursively(rewards, gamma, t=0):
    if t >= len(rewards):
        return 0
    return rewards[t] + gamma * calculate_value_recursively(rewards, gamma, t + 1)

# 계산
direct_value = calculate_value_directly(rewards, gamma)
recursive_value = calculate_value_recursively(rewards, gamma)

print(f"Direct Calculation: {direct_value}")
print(f"Recursive Calculation: {recursive_value}")
