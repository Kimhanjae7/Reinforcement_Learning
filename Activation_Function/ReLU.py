import numpy as np
import matplotlib.pyplot as plt

# ReLU function 정의
def relu(x):
    return np.maximum(0, x)

# -10 ~ +10 구간 500개 점 생성
x = np.linspace(-10, 10, 500)  
print(x)

# ReLU function으로 출력값 변환
y = relu(x)

# ReLU function 시각화
plt.figure(figsize=(8, 6))
plt.plot(x, y, label="ReLU Function", color="blue")
plt.title("ReLU Function", fontsize=16)
plt.xlabel("x", fontsize=14)
plt.ylabel("relu(x)", fontsize=14)
plt.axhline(0, color="red", linestyle="--", label="y = 0")
plt.axvline(0, color="green", linestyle="--", label="x = 0")
plt.legend(fontsize=12)
plt.grid(True)
plt.show()
