import numpy as np
import matplotlib.pyplot as plt

""" 
경사상승법에서 x는 최적화 대상
강화학습 및 최적화에서는 목표 함수를 최대화하는 방향으로 매개변수 x를 조정
최종 목표는 x 값을 조정하여 목표 함수의 값을 최대화하는 것

정책 신경망(Policy Network)의 학습 목표는 목표 함수(Objective Function)를 최대화하는 것. 이 목표 함수는 보통 장기 기대 보상
손실 함수처럼 최소화해야 할 값이 아니라, 보상을 많이 얻기 위해 정책 성능(목표 함수)을 최대화
-> 경사 상승법(Gradient Ascent)은 목표 함수를 최대화하기 위해, 목표 함수의 기울기(Gradient)를 따라 매개변수를 업데이트하는 최적화 알고리즘
"""

# 목표 함수 정의: f(x) = -x^2 + 4x (최대화하려는 함수)
def objective_function(x):
    return -x**2 + 4 * x

# 목표 함수의 기울기(미분): f'(x) = -2x + 4
def gradient(x):
    return -2 * x + 4

# 경사상승법
def gradient_ascent(initial_x, learning_rate, num_iterations):
    x = initial_x  # 초기값 설정
    history = []  # 각 반복에서 x 값을 저장하여 시각화를 위해 기록
    
    for i in range(num_iterations):  # 지정된 횟수만큼 반복
        grad = gradient(x)  # 현재 x에서의 기울기 계산
        x = x + learning_rate * grad  # 업데이트 (기울기 방향으로 이동)
        history.append(x)  # 현재 x 값을 기록
        print(f"Iteration {i+1}: x = {x:.5f}, objective = {objective_function(x):.5f}")
    
    return x, history  

initial_x = 0  # x의 초기값
learning_rate = 0.1  # 스텝 사이즈
num_iterations = 20  # 반복 횟수

final_x, history = gradient_ascent(initial_x, learning_rate, num_iterations)

# 경사상승법 과정 시각화를 위한 데이터 생성
x_values = np.linspace(-2, 4, 100)  # -2에서 4까지의 x 값 범위
y_values = objective_function(x_values)  # 각 x 값에 대한 목표 함수 값

# 경사상승법 과정 시각화
plt.figure(figsize=(8, 6))
plt.plot(x_values, y_values, label="Objective Function: f(x) = -x^2 + 4x", color="blue")
plt.scatter(history, [objective_function(h) for h in history], color="red", label="Gradient Ascent Steps") 
plt.title("Gradient Ascent on f(x) = -x^2 + 4x", fontsize=16) 
plt.xlabel("x", fontsize=14)
plt.ylabel("f(x)", fontsize=14)  
plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)  
plt.axvline(0, color="gray", linestyle="--", linewidth=0.8)  
plt.legend(fontsize=12)  
plt.grid(True)  
plt.show()  

