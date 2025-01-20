import numpy as np
import matplotlib.pyplot as plt

# 손실 함수 정의: f(x) = x^2
def loss_function(x):
    return x**2

# 손실 함수의 기울기(미분): f'(x) = 2x
def gradient(x):
    return 2 * x

# 경사하강법 
def gradient_descent(initial_x, learning_rate, num_iterations):
    x = initial_x  # 초기값 설정
    history = []  # 각 반복에서 x 값을 저장하여 시각화를 위해 기록
    
    for i in range(num_iterations):  # 지정된 횟수만큼 반복
        grad = gradient(x)  # 현재 x에서의 기울기 계산
        x = x - learning_rate * grad  # 업데이트 
        history.append(x)  # 현재 x 값을 기록
        print(f"Iteration {i+1}: x = {x:.5f}, loss = {loss_function(x):.5f}")
    
    return x, history  

initial_x = 10  # x의 초기값
learning_rate = 0.1  # 스텝 사이즈
num_iterations = 20  # 반복 횟수

final_x, history = gradient_descent(initial_x, learning_rate, num_iterations)

# 경사하강법 과정 시각화를 위한 데이터 생성
x_values = np.linspace(-10, 10, 100)  # -10에서 10까지의 x 값 범위
y_values = loss_function(x_values)  # 각 x 값에 대한 손실 함수 값

# 경사하강법 과정 시각화
plt.figure(figsize=(8, 6))
plt.plot(x_values, y_values, label="Loss Function: f(x) = x^2", color="blue")
plt.scatter(history, [loss_function(h) for h in history], color="red", label="Gradient Descent Steps") 
plt.title("Gradient Descent on f(x) = x^2", fontsize=16) 
plt.xlabel("x", fontsize=14)
plt.ylabel("f(x)", fontsize=14)  
plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)  
plt.axvline(0, color="gray", linestyle="--", linewidth=0.8)  
plt.legend(fontsize=12)  
plt.grid(True)  
plt.show()  
