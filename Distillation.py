import torch
import torch.nn as nn

# Teacher 모델 (큰 모델)
class TeacherModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 5)

    def forward(self, x):
        return self.layer(x)

# Student 모델 (작은 모델)
class StudentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 5)

    def forward(self, x):
        return self.layer(x)

# 증류 과정
def knowledge_distillation(teacher, student, data, alpha=0.5):
    loss_fn = nn.MSELoss()  # Teacher와 Student 간의 차이를 최소화
    teacher_outputs = teacher(data).detach()
    student_outputs = student(data)
    loss = alpha * loss_fn(student_outputs, teacher_outputs)
    return loss

# 실행
teacher = TeacherModel()
student = StudentModel()
data = torch.randn(10, 10)

loss = knowledge_distillation(teacher, student, data)
print("증류 손실:", loss.item())
