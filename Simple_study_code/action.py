import numpy as np
import random

# 간단한 행동 코드

# 환경: 행동 평가 함수
def evaluate_action(action):
    """
    주어진 행동에 따른 보상을 평가하는 함수.
    행동은 -1.0에서 +1.0 사이의 값 (연속적).
    """
    # 예시: 보상은 특정 조건에 따라 달라진다.
    return -abs(action - 0.5) + 1  # 0.5에 가까운 행동이 최적.

# 연속적 행동 공간에서 샘플링
def sample_actions(num_samples=10):
    """
    행동 공간에서 샘플링된 행동 반환.
    """
    return [random.uniform(-1.0, 1.0) for _ in range(num_samples)]

# 행동 확률 분포로 선택 (Softmax 기반)
def choose_action_with_probabilities():
    """
    행동 확률 분포를 기반으로 행동 선택.
    """
    action_probabilities = np.array([0.1, 0.7, 0.2])  # 행동 3개의 확률
    return np.random.choice([0, 1, 2], p=action_probabilities)

# 신경망 기반 행동 근사
def neural_network(state):
    """
    주어진 상태에 대해 각 행동의 Q값 반환 (예: 신경망 출력).
    """
    # 예제: 단순히 상태에 따라 Q값 반환.
    return np.array([0.1, 0.5, 0.3])

# 전체 실행 코드
if __name__ == "__main__":
    sampled_actions = sample_actions()
    best_sampled_action = max(sampled_actions, key=evaluate_action)
    print("1. 샘플링된 행동들:", sampled_actions)
    print("   샘플링 기반 최적 행동:", best_sampled_action)

    chosen_action = choose_action_with_probabilities()
    print("\n2. 확률 기반 선택된 행동:", chosen_action)

    current_state = [1, 0, 0]  # 예제 상태
    q_values = neural_network(current_state)
    best_nn_action = np.argmax(q_values)
    print("\n3. 신경망 기반 Q값:", q_values)
    print("   신경망 기반 최적 행동:", best_nn_action)
