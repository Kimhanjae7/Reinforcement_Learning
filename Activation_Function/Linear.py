# Linear 활성 함수의 간단한 파이썬 구현

import numpy as np
import matplotlib.pyplot as plt

"""
딥살사(Deep SARSA)에서 출력층이 선형함수인 이유

* Q-값은 연속적이고 제한되지 않은 값
    Q-값은 상태와 행동에 대한 누적 보상을 나타내므로, 이론적으로 값의 범위에 제한이 없음
    예: 매우 높은 보상 +∞+∞ 또는 매우 낮은 보상 −∞−∞로 나타날 수 있음
* 선형 활성 함수는 입력 값을 그대로 출력하므로, Q-값의 범위를 제한하지 않음
    f(x)=xf(x)=x: 입력 범위가 제한되지 않음

만약 다른 활성 함수를 사용하면?
    Sigmoid 함수:
        출력값이 0에서 1 사이로 제한되므로, Q-값의 표현력이 손실됨.
    Tanh 함수:
        출력값이 -1에서 1 사이로 제한되므로, 큰 보상이나 벌점을 제대로 표현할 수 없음.
"""
# Linear 활성 함수 정의
def linear_activation(x):
    """
    Linear 활성 함수
    :param x: 입력값 (스칼라 또는 배열)
    :return: 입력값 그대로 반환
    """
    return x

x = np.linspace(-10, 10, 100)  # -10에서 10까지 100개의 값 생성
# print(x)

y = linear_activation(x)

# 그래프 시각화
plt.figure(figsize=(8, 6))
plt.plot(x, y, label="Linear Activation: f(x) = x", color="blue")
plt.title("Linear Activation Function", fontsize=16)
plt.xlabel("Input (x)", fontsize=14)
plt.ylabel("Output (f(x))", fontsize=14)
plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)  
plt.axvline(0, color="gray", linestyle="--", linewidth=0.8)  
plt.grid(True)
plt.legend(fontsize=12)
plt.show()

