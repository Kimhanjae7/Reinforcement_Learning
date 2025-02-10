import gym
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from collections import deque


if not hasattr(np, 'bool'):
    np.bool = bool  # np.bool이 없으면 bool로 대체

# 자율주행 환경 설정 (CarRacing 환경 사용)
env = gym.make("CarRacing-v2", render_mode="human")

# 상태와 행동 개수 정의
state_size = env.observation_space.shape
action_size = env.action_space.shape[0]  # 연속형 행동 공간에서는 .shape[0] 사용

# 경험 저장을 위한 버퍼
memory = deque(maxlen=2000)

# 하이퍼파라미터 설정
gamma = 0.95        # 할인율 (미래 보상 반영)
epsilon = 1.0       # 탐험률 (랜덤 행동 비율)
epsilon_min = 0.01  # 탐험 최소값
epsilon_decay = 0.995  # 탐험률 감소율
batch_size = 32     # 학습할 데이터 개수

# DQN 모델 정의
def build_model():
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=state_size),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(action_size, activation='linear')  # 연속형 행동 공간에서는 선형 출력 사용
    ])
    model.compile(loss="mse", optimizer=keras.optimizers.Adam(learning_rate=0.001))
    return model

# 모델 생성
model = build_model()

# DQN 학습 함수
def train():
    global epsilon
    print("🔹 Training started...", flush=True)

    for episode in range(1000):  # 1000번 학습
        print(f"🔹 Episode {episode} started (epsilon: {epsilon:.4f})", flush=True)
        state, _ = env.reset()
        state = np.reshape(state, [1, *state_size])

        for time in range(500):  # 최대 500 프레임 실행
            # 행동 선택 (탐험 vs. 활용)
            if np.random.rand() <= epsilon:
                action = env.action_space.sample()  # 랜덤 행동 선택
            else:
                q_values = model.predict(state, verbose=0)
                action = np.tanh(q_values[0])  # 행동을 [-1, 1] 범위로 정규화
                action = np.clip(action, -1, 1)  # 안전한 범위 유지

            # 행동 수행 및 보상 확인
            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated
            next_state = np.reshape(next_state, [1, *state_size])

            # 경험 저장
            memory.append((state, action, reward, next_state, done))
            state = next_state

            if done:
                print(f" Episode {episode} finished - Score: {time}, Epsilon: {epsilon:.4f}", flush=True)
                break

        # 경험 학습 (Replay)
        if len(memory) > batch_size:
            minibatch = random.sample(memory, batch_size)
            for state, action, reward, next_state, done in minibatch:
                target = reward
                if not done:
                    target = reward + gamma * np.max(model.predict(next_state, verbose=0)[0])
                target_f = model.predict(state, verbose=0)
                target_f[0] = target  # 연속형 행동 공간에서는 전체 값을 업데이트
                model.fit(state, target_f, epochs=1, verbose=0)

        # 탐험률 감소
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

# 학습 시작
if __name__ == "__main__":
    train()
