import matplotlib.pyplot as plt
import numpy as np

"""
정책 신경망의 출력은 에이전트가 가능한 행동들 중에서 어떤 행동을 선택할 확률.
Softmax는 출력값을 확률 값으로 변환하여, 행동 선택이 확률적으로 이루어지게 함.
예)
    네트워크 출력: z=[2.0,1.0,0.1]
    Softmax 출력: [0.70,0.21,0.09] → 행동 1, 2, 3의 선택 확률.

정책신경망에서는 입력이 상태, 출력이 각 행동을 할 확률
즉, 정책의 정의가 각 행동을 할 확률이기 때문에 이 확률을 모두 합하면 1이 됨. -> Softmax또한 출력을 모두 더하면 1이됨
"""
# Softmax 함수 정의
def softmax(logits):
    """
    Softmax 함수: 입력값을 확률 분포로 변환
    :param logits: 입력 배열 (로짓 값들)
    :return: 입력값에 대한 Softmax 확률 분포
    """
    exp_values = np.exp(logits - np.max(logits)) 
    probabilities = exp_values / np.sum(exp_values)
    return probabilities

# 로짓 값 (Softmax에 입력될 값)
logits = [3.0, 2.0, 1.0, 0.5]

probabilities = softmax(logits)

# 시각화
labels = [f"Action {i+1}" for i in range(len(logits))] 
plt.figure(figsize=(8, 8))
plt.pie(probabilities, labels=labels, autopct='%1.2f%%', shadow=True, startangle=90)
plt.title("Softmax Output as Pie Chart", fontsize=16)
plt.show()
