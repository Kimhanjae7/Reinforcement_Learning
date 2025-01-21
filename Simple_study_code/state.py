# 상태 설정의 중요성

class GoodState:
    def __init__(self, position, speed, traffic_light, distance_to_car):
        self.position = position
        self.speed = speed
        self.traffic_light = traffic_light
        self.distance_to_car = distance_to_car

class BadState:
    def __init__(self, speed):
        self.speed = speed  # 속도만 포함

# 자율주행 에이전트
class AutonomousCar:
    def decide_action(self, state):
        if isinstance(state, GoodState):  # 좋은 상태 기반 행동
            if state.traffic_light == "red":
                return "Stop"
            elif state.distance_to_car < 10:
                return "Slow down"
            else:
                return "Maintain speed"
        elif isinstance(state, BadState):  # 나쁜 상태 기반 행동
            # 정보 부족으로 인해 단순 행동만 수행 가능
            return "Cannot decide (insufficient information)"

# 상태 설정
good_state = GoodState(
    position=(50, 100),
    speed=60,
    traffic_light="red",  # 빨간불
    distance_to_car=5      # 앞차와의 거리
)

bad_state = BadState(
    speed=60  # 속도만 있음
)

# 에이전트 동작 테스트
car = AutonomousCar()

print("Using GoodState:")
print(f"Action: {car.decide_action(good_state)}")  # 좋은 상태를 이용한 결과

print("\nUsing BadState:")
print(f"Action: {car.decide_action(bad_state)}")  # 나쁜 상태를 이용한 결과
