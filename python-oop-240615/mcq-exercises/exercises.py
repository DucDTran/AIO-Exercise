from abc import ABC, abstractmethod
import torch
import torch .nn as nn
import datetime


# question 1


data = torch . Tensor([1, 2, 3])
softmax_function = nn. Softmax(dim=0)
output = softmax_function(data)
# assert round(output[0]. item(), 2) == 0.09
print(output)

# question 2


class MySoftmax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x_exp = torch.exp(x)
        total = x_exp.sum(0, keepdims=True)
        return x_exp / total


data = torch . Tensor([5, 2, 4])
my_softmax = MySoftmax()
output = my_softmax(data)
# assert round(output[-1]. item(), 2) == 0.26
print(output)


# question 3
class MySoftmax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x_exp = torch.exp(x)
        total = x_exp.sum(0, keepdims=True)
        return x_exp / total


data = torch . Tensor([1, 2, 300000000])
my_softmax = MySoftmax()
output = my_softmax(data)
# assert round(output[0]. item(), 2) == 0.0
print(output)

# question 4


class SoftmaxStable(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x_max = torch.max(x, dim=0)
        x_exp = torch.exp(x - x_max.values)
        partition = x_exp .sum(0, keepdims=True)
        return x_exp / partition


data = torch.Tensor([1, 2, 3])
softmax_stable = SoftmaxStable()
output = softmax_stable(data)
# assert round(output[-1]. item(), 2) == 0.67
print(output)


# question 5


class Person(ABC):
    def __init__(self, name, yob):
        self._name = name
        self._yob = yob

    @abstractmethod
    def describe(self):
        pass

    def get_age(self):
        return datetime.date.today().year - self._yob

    def get_year(self):
        return self._yob


class Student(Person):
    def __init__(self, name, yob, grade):
        super().__init__(name=name, yob=yob)
        self.__grade = grade

    def describe(self):
        print(
            f'Student - Name : {self._name} - YoB: {self._yob} - Grade: {self.__grade}')


student1 = Student(name=" studentZ2023 ", yob=2011, grade="6")
# assert student1._yob == 2011
# student1.describe()

# question 6


class Teacher(Person):
    def __init__(self, name, yob, subject):
        super().__init__(name=name, yob=yob)
        self.__subject = subject

    def describe(self):
        print(
            f'Teacher - Name : {self._name} - YoB: {self._yob} - Subject: {self.__subject}')


teacher1 = Teacher(name=" teacherZ2023 ", yob=1991, subject=" History ")
# assert teacher1._yob == 1991
# teacher1.describe()

# question 7


class Doctor(Person):
    def __init__(self, name, yob, specialist):
        super().__init__(name=name, yob=yob)
        self.__specialist = specialist

    def describe(self):
        print(
            f'Doctor - Name : {self._name} - YoB: {self._yob} - Specialist: {self.__specialist}')


doctor1 = Doctor(name=" doctorZ2023 ", yob=1981,
                 specialist=" Endocrinologists ")
# assert doctor1._yob == 1981
# doctor1.describe()

# question 8


class Ward:
    def __init__(self, name):
        self.__name = name
        self.__people_list = []

    def add_person(self, person: Person):
        self.__people_list.append(person)

    def describe(self):
        print(f'Ward Name: {self.__name}')
        for person in self.__people_list:
            person.describe()

    def count_doctor(self):
        count = 0
        for person in self.__people_list:
            if isinstance(person, Doctor):
                count += 1
        return count


ward1 = Ward(name="Ward1")
student1 = Student(name="studentA", yob=2010, grade="7")
teacher1 = Teacher(name="teacherA", yob=1969, subject="Math")
teacher2 = Teacher(name="teacherB", yob=1995, subject="History")
doctor1 = Doctor(name="doctorA", yob=1945, specialist="Endocrinologists")
doctor2 = Doctor(name="doctorB", yob=1975, specialist="Cardiologists")

ward1.add_person(student1)
ward1.add_person(teacher1)
ward1.add_person(teacher2)
ward1.add_person(doctor1)
ward1.add_person(doctor2)
print(ward1.count_doctor())

# question 9


class MyStack:
    def __init__(self, capacity):
        self.__capacity = capacity
        self.__stack = []

    def is_empty(self):
        return len(self.__stack) == 0

    def is_full(self):
        return len(self.__stack) == self.__capacity

    def push(self, value):
        if self.is_full():
            print('Stack Overflow!')
        else:
            return self.__stack.append(value)

    def pop(self):
        if self.is_empty():
            print('Stack Underflow!')
        else:
            return self.__stack.pop()

    def top(self):
        if self.is_empty():
            print('Stack Underflow')
        else:
            return self.__stack[-1]


stack1 = MyStack(capacity=5)
stack1.push(1)
assert stack1.is_full() == False
stack1.push(2)
print(stack1.is_full())

# question 10


stack1 = MyStack(capacity=5)
stack1.push(1)
assert stack1.is_full() == False
stack1.push(2)
print(stack1.top())


# question 11

class MyQueue:
    def __init__(self, capacity):
        self.__capacity = capacity
        self.__queue = []

    def is_empty(self):
        return len(self.__queue) == 0

    def is_full(self):
        return len(self.__queue) == self.__capacity

    def enqueue(self, value):
        if self.is_full():
            print('Stack Overflow!')
        return self.__queue.append(value)

    def dequeue(self):
        if self.is_empty():
            print('Queue is empty!')
        return self.__queue.pop(0)

    def front(self):
        if self.is_empty():
            print('Queue is empty!')
            return
        return self.__queue[0]


queue1 = MyQueue(capacity=5)
queue1.enqueue(1)
assert queue1.is_full() == False
queue1.enqueue(2)
print(queue1.is_full())

# question 12


queue1 = MyQueue(capacity=5)
queue1.enqueue(1)
assert queue1.is_full() == False
queue1.enqueue(2)
print(queue1.front())
