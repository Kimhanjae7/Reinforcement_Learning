import gym
import numpy as np

if not hasattr(np, 'bool'):
    np.bool = bool  # numpy.bool이 없으면 bool로 대체

import torch
from stable_baselines3 import PPO

# CarRacing 환경 생성 (화면 렌더링 설정)
env = gym.make("CarRacing-v2", render_mode="human")

# PPO 모델 생성 (CNN 정책 사용)
model = PPO("CnnPolicy", env, verbose=1, tensorboard_log="./ppo_carracing_tensorboard/")

# 학습 하이퍼파라미터 설정
total_episodes = 100  # 총 학습 에피소드 수
timesteps_per_episode = 1000  # 한 에피소드당 최대 실행 시간

# 학습 시작
print("🔹 Training started...")
for episode in range(total_episodes):
    print(f"🔹 Episode {episode+1} started")  # 에피소드 시작 로그
    obs, _ = env.reset()
    total_reward = 0  # 에피소드 동안 받은 총 보상

    for step in range(timesteps_per_episode):
        env.render()  # 화면 출력

        # PPO 모델을 사용하여 최적 행동 선택
        action, _ = model.predict(obs)

        # 환경 실행 및 보상 획득
        obs, reward, done, truncated, _ = env.step(action)
        total_reward += reward  # 보상 누적

        if done or truncated:
            break  # 에피소드 종료

    # 에피소드 결과 출력
    print(f"   →  Episode {episode+1} finished - Score: {total_reward:.2f}")

# 학습 완료 후 저장
model.save("ppo_carracing_model")
print("✅ Training finished! Model saved as 'ppo_carracing_model'.")

# 학습된 모델 실행 (자동차 주행)
print("\n🔹 Running trained model...\n")
obs, _ = env.reset()
while True:
    env.render()
    action, _ = model.predict(obs)  # 최적 행동 예측
    obs, reward, done, truncated, _ = env.step(action)
    if done or truncated:
        obs, _ = env.reset()  # 다시 시작
