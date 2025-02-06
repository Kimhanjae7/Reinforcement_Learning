import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 1️⃣ 훈련 데이터 생성
x_train = np.array([1, 2, 3, 4, 5], dtype=np.float32)
y_train = np.array([2, 4, 6, 8, 10], dtype=np.float32)

# 2️⃣ 가중치와 편향 변수 정의
weight = tf.Variable(0.0)  # 초기 가중치 (W)
bias = tf.Variable(0.0)    # 초기 편향 (b)

# 3️⃣ 모델 정의 (선형 함수: y = W*x + b)
def linear_regression(x):
    return weight * x + bias

# 4️⃣ 손실 함수 정의 (평균 제곱 오차: MSE)
def loss_function(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

# 5️⃣ 최적화 알고리즘 선택 (경사 하강법)
optimizer = tf.optimizers.SGD(learning_rate=0.01)

# 6️⃣ 훈련 함수 정의
def train_step(x, y):
    with tf.GradientTape() as tape:
        # 예측 값 계산
        y_pred = linear_regression(x)
        # 손실(오차) 계산
        loss_value = loss_function(y, y_pred)
    
    # 가중치(weight)와 편향(bias)에 대한 기울기 계산
    gradients = tape.gradient(loss_value, [weight, bias])
    
    # 가중치와 편향을 업데이트 (최적화 수행)
    optimizer.apply_gradients(zip(gradients, [weight, bias]))
    
    return loss_value

# 7️⃣ 훈련 (1000번 반복)
epochs = 1000
for epoch in range(epochs):
    loss = train_step(x_train, y_train)

    # 100번마다 손실(loss) 출력
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}: Loss = {loss.numpy():.4f}, Weight = {weight.numpy():.4f}, Bias = {bias.numpy():.4f}")

# 8️⃣ 학습된 모델로 예측 수행
x_test = np.array([6, 7, 8, 9, 10], dtype=np.float32)
y_pred = linear_regression(x_test)

# 9️⃣ 결과 출력 및 그래프 시각화
print("예측 결과:", y_pred.numpy())

plt.scatter(x_train, y_train, label="Train Data", color="blue")  # 실제 데이터
plt.plot(x_test, y_pred, label="Model Prediction", color="red")  # 예측 결과
plt.xlabel("X 값")
plt.ylabel("Y 값")
plt.legend()
plt.title("선형 회귀 학습 결과")
plt.show()
