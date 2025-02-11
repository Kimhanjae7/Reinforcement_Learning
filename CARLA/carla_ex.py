#!/usr/bin/env python3
import gym
import carla
import gym_carla
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import time

# 1️⃣ Carla 서버 실행


# 2️⃣ 환경 생성 (Carla를 Gym 환경으로 변환)
env = make_vec_env(lambda: gym.make('CarlaEnv-v0'), n_envs=4)

# 3️⃣ 강화학습 모델 정의
model = PPO("MlpPolicy", env, verbose=1)

# 4️⃣ 학습 실행
model.learn(total_timesteps=100000)

# 5️⃣ 모델 저장
model.save("ppo_carla")

# 6️⃣ 학습된 모델 테스트
obs = env.reset()
while True:
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render()  # 화면 출력 (옵션)
    if done:
        break
env.close()

print("✅ 학습 완료! 모델 저장됨.")
