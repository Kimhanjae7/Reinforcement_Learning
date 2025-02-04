import subprocess

# 규칙: AI가 계산한 답과 실제 수학적 결과 비교
def verify_math_solution(answer, correct_solution):
    return answer == correct_solution

# 규칙: Python 코드가 정상적으로 실행되는지 확인
def verify_code_execution(code):
    try:
        exec(code)  # 코드 실행
        return True
    except Exception as e:
        print("❌ 코드 실행 오류:", e)
        return False
    


# AI 모델이 생성한 답
ai_answer = 42
correct_answer = 6 * 7  # 실제 올바른 정답

# 검증 수행
if verify_math_solution(ai_answer, correct_answer):
    print("✅ AI 답변이 올바릅니다!")
else:
    print("❌ AI 답변이 틀렸습니다.")


print("--------------------------------------")
# AI가 생성한 코드 예제
ai_generated_code = """
def add(a, b):
    return a + b
"""

# 검증 수행
if verify_code_execution(ai_generated_code):
    print("✅ 코드가 정상적으로 실행됩니다!")
else:
    print("❌ 코드 실행에 오류가 있습니다.")
