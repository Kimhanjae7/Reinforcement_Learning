states = ["S1", "S2", "S3"]
actions = ["A1", "A2"]
transition_probabilities = {
    ("S1", "A1", "S2"): 0.8,
    ("S1", "A1", "S3"): 0.2,
    ("S1", "A2", "S2"): 0.5,
    ("S1", "A2", "S3"): 0.5,
    ("S2", "A1", "S3"): 1.0,
    ("S2", "A2", "S1"): 1.0,
    ("S3", "A1", "S1"): 0.3,
    ("S3", "A1", "S2"): 0.7,
    ("S3", "A2", "S1"): 0.6,
    ("S3", "A2", "S2"): 0.4,
}
rewards = {
    ("S1", "A1", "S2"): 5,
    ("S1", "A1", "S3"): 1,
    ("S1", "A2", "S2"): 0,
    ("S1", "A2", "S3"): 2,
    ("S2", "A1", "S3"): 10,
    ("S2", "A2", "S1"): -1,
    ("S3", "A1", "S1"): 4,
    ("S3", "A1", "S2"): 6,
    ("S3", "A2", "S1"): 3,
    ("S3", "A2", "S2"): 8,
}
gamma = 0.9  # 감가율
policy = {
    "S1": {"A1": 0.7, "A2": 0.3},
    "S2": {"A1": 0.6, "A2": 0.4},
    "S3": {"A1": 0.5, "A2": 0.5},
}

# 상태 가치 함수 계산
def calculate_state_value(V, Q):
    new_V = {}
    for state in states:
        value = 0
        for action, action_prob in policy[state].items():
            value += action_prob * Q[state][action]
        new_V[state] = value
    return new_V

# 행동 가치 함수 계산
def calculate_action_value(V):
    Q = {}
    for state in states:
        Q[state] = {}
        for action in actions:
            value = 0
            for next_state in states:
                prob = transition_probabilities.get((state, action, next_state), 0)
                reward = rewards.get((state, action, next_state), 0)
                value += prob * (reward + gamma * V.get(next_state, 0))
            Q[state][action] = value
    return Q

# 초기화 및 계산
V = {state: 0 for state in states}  # 초기 상태 가치 함수
Q = {state: {action: 0 for action in actions} for state in states}  # 초기 행동 가치 함수
iterations = 10  # 반복 횟수

print("Value and Action Value Function Updates:")
for i in range(iterations):
    print(f"Iteration {i + 1}:")
    
    # 행동 가치 함수 업데이트
    Q = calculate_action_value(V)
    print("  Action Value Function:")
    for state, actions in Q.items():
        for action, value in actions.items():
            print(f"    Q({state}, {action}): {value:.4f}")
    
    # 상태 가치 함수 업데이트
    V = calculate_state_value(V, Q)
    print("  State Value Function:")
    for state, value in V.items():
        print(f"    V({state}): {value:.4f}")
    
    print("------------------------------------")

# 최종 결과 출력
print("\nFinal State Value Function:")
for state, value in V.items():
    print(f"{state}: {value:.4f}")

print("\nFinal Action Value Function:")
for state, actions in Q.items():
    for action, value in actions.items():
        print(f"{state}, {action}: {value:.4f}")
