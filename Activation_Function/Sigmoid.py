import numpy as np
import matplotlib.pyplot as plt

# Sigmoid function 정의
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# -10 ~ +10 구간 500개 점 생성
x = np.linspace(-10, 10, 500)  
print(x)

# Sigmoid function으로 출력값 변환
y = sigmoid(x)
print(y)

# Sigmoid function 시각화
plt.figure(figsize=(8, 6))
plt.plot(x, y, label="Sigmoid Function", color="blue")
plt.title("Sigmoid Function", fontsize=16)
plt.xlabel("x", fontsize=14)
plt.ylabel("sigmoid(x)", fontsize=14)
plt.axhline(0.5, color="red", linestyle="--", label="y = 0.5")
plt.axvline(0, color="green", linestyle="--", label="x = 0")
plt.legend(fontsize=12)
plt.grid(True)
plt.show()
