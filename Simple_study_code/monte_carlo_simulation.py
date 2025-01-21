import matplotlib.pyplot as plt

def update_value(value, target, learning_rate):
    """
    가치 함수를 업데이트하는 함수
    :param value: 현재 상태 가치 함수 V(s)
    :param target: 실제 보상 또는 목표값 G(s)
    :param learning_rate: 학습 속도 (1/n)
    :return: 업데이트된 가치 함수
    """
    error = target - value  # 오차 계산
    updated_value = value + learning_rate * error  # 가치 함수 업데이트
    print(f"Current Value: {value:.4f}, Target: {target:.4f}, Error: {error:.4f}, Updated Value: {updated_value:.4f}")
    return updated_value


def simulate_value_update_with_plot(initial_value, targets, learning_rate):
    """
    학습 과정을 그래프로 시각화
    :param initial_value: 초기 가치 함수 값
    :param targets: 실제 보상의 리스트 (G(s))
    :param learning_rate: 학습 속도 (1/n)
    """
    values = [initial_value]
    value = initial_value

    print("=== Value Update Simulation ===")
    for i, target in enumerate(targets):
        print(f"\nStep {i + 1}:")
        value = update_value(value, target, learning_rate)
        values.append(value)

    # 그래프 시각화
    steps = range(len(values))
    plt.plot(steps, values, marker='o', label="V(s)")
    plt.axhline(y=targets[-1], color='r', linestyle='--', label="Target (G(s))")
    plt.title("Value Update Over Time")
    plt.xlabel("Steps")
    plt.ylabel("Value")
    plt.legend()
    plt.grid()
    plt.show()


# 초기 값
initial_value = 5.0  # 초기 상태 가치 함수 V(s)
targets = [9.0, 8.5, 8.0, 8.8, 9.2]  # 실제 보상 (G(s)) 리스트
learning_rate = 0.5  # 학습 속도 (1/n)

# 시뮬레이션 실행 및 그래프 출력
simulate_value_update_with_plot(initial_value, targets, learning_rate)
