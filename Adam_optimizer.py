import numpy as np
import matplotlib.pyplot as plt
"""
특성	    Adam Optimizer            |    표준 경사하강법
학습률	    매개변수마다 동적으로 조정	|    고정된 학습률 사용
수렴 속도	더 빠르게 수렴	           |   비교적 느림
안정성	    기울기 진동 및 발산 방지    |    기울기 진동이 발생할 수 있음
복잡성    	계산이 약간 더 복잡	       |     계산이 단순

"""

# 손실 함수 정의: f(x) = x^2
def loss_function(x):
    return x**2

# 손실 함수의 기울기(미분): f'(x) = 2x
def gradient(x):
    return 2 * x

# Adam Optimizer
def adam_optimizer(initial_x, learning_rate, beta1, beta2, epsilon, num_iterations):
    x = initial_x  # 초기값 설정
    m = 0  # 1차 모멘트 초기화(평균)
    v = 0  # 2차 모멘트 초기화(분산)
    history = []  # x 값의 기록
    
    for t in range(1, num_iterations + 1):
        g = gradient(x)  # 현재 x에서의 기울기 계산
        
        # 1차 모멘트 추정
        m = beta1 * m + (1 - beta1) * g
        
        # 2차 모멘트 추정
        v = beta2 * v + (1 - beta2) * (g ** 2)
        
        # 바이어스 보정
        m_hat = m / (1 - beta1**t)  # 1차 모멘트 보정
        v_hat = v / (1 - beta2**t)  # 2차 모멘트 보정
        
        # 매개변수 업데이트
        x = x - (learning_rate * m_hat) / (np.sqrt(v_hat) + epsilon)
        
        # 현재 x 값을 기록
        history.append(x)
        print(f"Iteration {t}: x = {x:.5f}, loss = {loss_function(x):.5f}")
    
    return x, history

# Adam Optimizer 파라미터 설정
initial_x = 10  # x의 초기값
learning_rate = 0.5  # 학습률
beta1 = 0.9  # 1차 모멘텀 감쇠율
beta2 = 0.999  # 2차 모멘텀 감쇠율
epsilon = 1e-8  # 작은 값 (0으로 나누는 문제 방지)
num_iterations = 20  # 반복 횟수

final_x, history = adam_optimizer(initial_x, learning_rate, beta1, beta2, epsilon, num_iterations)

# 시각화를 위한 데이터 생성
x_values = np.linspace(-10, 10, 100)  # -10에서 10까지의 x 값 범위
y_values = loss_function(x_values)  # 각 x 값에 대한 손실 함수 값

# Adam Optimizer 과정 시각화
plt.figure(figsize=(8, 6))
plt.plot(x_values, y_values, label="Loss Function: f(x) = x^2", color="blue")
plt.scatter(history, [loss_function(h) for h in history], color="red", label="Adam Steps") 
plt.title("Adam Optimization on f(x) = x^2", fontsize=16) 
plt.xlabel("x", fontsize=14)
plt.ylabel("f(x)", fontsize=14)  
plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)  
plt.axvline(0, color="gray", linestyle="--", linewidth=0.8)  
plt.legend(fontsize=12)  
plt.grid(True)  
plt.show()

