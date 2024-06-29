from abc import ABC, abstractmethod
import datetime


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


class Doctor(Person):
    def __init__(self, name, yob, specialist):
        super().__init__(name=name, yob=yob)
        self.__specialist = specialist

    def describe(self):
        print(
            f'Doctor - Name : {self._name} - YoB: {self._yob} - Specialist: {self.__specialist}')


class Teacher(Person):
    def __init__(self, name, yob, subject):
        super().__init__(name=name, yob=yob)
        self.__subject = subject

    def describe(self):
        print(
            f'Teacher - Name : {self._name} - YoB: {self._yob} - Subject: {self.__subject}')


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

    def sort_age(self):
        self.__people_list.sort(key=lambda x: x.get_age())

    def compute_average(self):
        total = 0
        teacher_count = 0
        for person in self.__people_list:
            if isinstance(person, Teacher):
                teacher_count += 1
                total += person.get_year()
        return total / teacher_count


student1 = Student(name='studentA', yob=2010, grade='7')
student1.describe()
teacher1 = Teacher(name='teacherA', yob=1969, subject='Math')
teacher1.describe()
doctor1 = Doctor(name='doctorA', yob=1945, specialist='Endocrinologists')
doctor1.describe()

teacher2 = Teacher(name=" teacherB ", yob=1995, subject=" History ")
doctor2 = Doctor(name=" doctorB ", yob=1975, specialist=" Cardiologists ")
ward1 = Ward(name=" Ward1 ")
ward1.add_person(student1)
ward1.add_person(teacher1)
ward1.add_person(teacher2)
ward1.add_person(doctor1)
ward1.add_person(doctor2)
ward1.describe()

print(f'\nNumber of doctors: {ward1.count_doctor()}')

print('\nAfter sorting Age of Ward1 people')
ward1.sort_age()
ward1.describe()

print(f'\nAverage year of birth (teachers): {ward1.compute_average()}')
