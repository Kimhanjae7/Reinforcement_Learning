import numpy as np

# 상태 및 보상 설정
states = ["Intersection", "Highway", "Destination"]
rewards = {"Intersection": -5, "Highway": 10, "Destination": 100}
transition_probabilities = {
    ("Intersection", "Highway"): 0.9,
    ("Intersection", "Destination"): 0.1,
    ("Highway", "Destination"): 1.0,
}

# 감가율 설정
gamma = 0.9

# 벨만 방정식 기반 가치 계산
def calculate_value(state, gamma):
    if state == "Destination":
        return rewards[state]
    value = 0
    for next_state, prob in transition_probabilities.items():
        if next_state[0] == state:
            value += prob * (rewards[next_state[1]] + gamma * calculate_value(next_state[1], gamma))
    return value

# 각 상태의 가치 계산
for state in states:
    value = calculate_value(state, gamma)
    print(f"State: {state}, Value: {value:.2f}")
