# MSE (Mean Squared Error, 평균제곱오차) 계산 간단한 파이썬 코드

# 실제값과 예측값의 리스트
actual_values = [3, -0.5, 2, 7]  # 실제값 (y)
predicted_values = [2.5, 0.0, 2, 8]  # 예측값 (ŷ)

"""
    <제곱 이유>
    실제값과 예측값의 차이는 음수나 양수일 수 있음
    모든 오차를 더할 때, 음수와 양수가 서로 상쇄될 수 있음

    예: (-2)+2=0→ 오차가 실제로 존재하지만 0으로 계산됨

    -> 제곱을 하면 음수 오차도 양수로 변환되므로, 모든 오차를 제대로 반영가능

    * MSE는 실제값과 예측값 간의 차이를 제곱하여 평균낸 값
    * 값이 작을수록 예측이 실제값에 더 가깝다는 것을 의미
"""

# MSE 계산 함수
"""
MSE(Mean Squared Error)를 계산하는 함수
    :param actual: 실제값 
    :param predicted: 예측값 
    :return: MSE 값
"""
def calculate_mse(actual, predicted):
    # 실제값과 예측값의 차이의 제곱을 합산
    squared_errors = [(a - p) ** 2 for a, p in zip(actual, predicted)]
    # 제곱오차의 평균 계산
    mse = sum(squared_errors) / len(actual)
    return mse

mse = calculate_mse(actual_values, predicted_values)

# 결과 출력
print(f"실제값: {actual_values}")
print(f"예측값: {predicted_values}")
print(f"평균제곱오차 (MSE): {mse:.4f}")


