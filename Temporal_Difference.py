import numpy as np

# 환경 설정
states = [0, 1, 2, 3, 4]  # 상태 공간
terminal_state = 4         # 종료 상태
gamma = 0.9                # 할인율 (discount factor)
alpha = 0.1                # 학습률 (learning rate)
episodes = 1000            # 학습할 에피소드 수

# 초기 상태 가치 함수 (모든 상태를 0으로 초기화)
V = np.zeros(len(states))

# TD(0) 학습 시작
for episode in range(episodes):
    state = np.random.choice(states[:-1])  # 종료 상태 제외한 랜덤 시작
    while state != terminal_state:
        # 행동 선택 (무작위로 왼쪽 or 오른쪽 이동)
        next_state = state + np.random.choice([-1, 1])
        next_state = max(0, min(next_state, terminal_state))  # 범위 제한

        # 보상 설정 (이동 시 -1)
        reward = -1

        # TD(0) 업데이트 공식 적용
        V[state] += alpha * (reward + gamma * V[next_state] - V[state])

        # 다음 상태로 이동
        state = next_state

# 결과 출력
print("학습된 상태 가치 함수:")
for s in states:
    print(f"V({s}) = {V[s]:.4f}")
