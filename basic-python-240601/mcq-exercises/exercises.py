# question_1
import numpy as np
import math


def calc_f1_score(tp, fp, fn):

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * ((precision * recall) / (precision + recall))

    return f1_score


assert (np.isclose(round(calc_f1_score(tp=2, fp=3, fn=5), 2),
        0.33, rtol=1e-09, atol=1e-09))
print(round(calc_f1_score(tp=2, fp=4, fn=5), 2))


# question_2


def is_number(n):
    try:
        float(n)
    except ValueError:
        return False
    return True


assert (np.isclose(is_number(3), 1.0, rtol=1e-09, atol=1e-09))
assert (np.isclose(is_number('-2a'), 0.0, rtol=1e-09, atol=1e-09))
print(is_number(1))
print(is_number('n'))

# question 3
# RELU


# question 4


def calc_sig(x):
    return 1 / (1 + np.exp(-x))


assert (np.isclose(round(calc_sig(3), 2), 0.95, rtol=1e-09, atol=1e-09))
print(round(calc_sig(2), 2))

# question 5


def calc_elu(x):
    elu_value = 0.01 * math.exp(x) - 1 if x <= 0 else x
    return elu_value


assert (round(calc_elu(1))) == 1
print(round(calc_elu(-1), 2))

# question 6


def sigmoid_func(x):
    return 1 / (1 + np.exp(-x))


def relu_func(x):
    return 0 if x <= 0 else x


def elu_func(x):
    return 0.01 * (math.exp(x) - 1) if x <= 0 else x


def calc_activation_func(x, act_name):
    if act_name == 'sigmoid':
        return sigmoid_func(x)
    elif act_name == 'relu':
        return relu_func(x)
    elif act_name == 'elu':
        return elu_func(x)


assert calc_activation_func(x=1, act_name='relu') == 1
print(round(calc_activation_func(x=3, act_name='sigmoid'), 2))

# question 7


def calc_ae(y, y_hat):
    value = y - y_hat
    return value if value >= 0 else value * -1


assert calc_ae(1, 6) == 5
print(calc_ae(2, 9))

# question 8


def calc_se(y, y_hat):
    return (y - y_hat) ** 2


assert calc_se(4, 2) == 4
print(calc_se(2, 1))

# question 9


def calc_factorial(n):
    if (n == 0) or (n == 1):
        return 1
    else:
        return n * calc_factorial(n - 1)


def approx_cos(x, n):
    cos_value = 0
    for i in range(0, n):
        value = np.power(-1, i) * (np.power(x, (2*i)) / calc_factorial((2*i)))
        cos_value += value
    return cos_value


assert np.isclose(round(approx_cos(x=1, n=10), 2),
                  0.54, rtol=1e-09, atol=1e-09)
print(round(approx_cos(x=3.14, n=10), 2))

# question 10


def approx_sin(x, n):
    sin_value = 0
    for i in range(0, n):
        value = np.power(-1, i) * (np.power(x, (2*i+1)) /
                                   calc_factorial((2*i+1)))
        sin_value += value
    return sin_value


assert np.isclose(round(approx_sin(x=1, n=10), 4),
                  0.8415, rtol=1e-09, atol=1e-09)
print(round(approx_sin(x=3.14, n=10), 4))

# question 11


def approx_sinh(x, n):
    sinh_value = 0
    for i in range(0, n):
        value = (np.power(x, (2*i+1)) / calc_factorial((2*i+1)))
        sinh_value += value
    return sinh_value


assert np.isclose(round(approx_sinh(x=1, n=10), 2),
                  1.18, rtol=1e-09, atol=1e-09)
print(round(approx_sinh(x=3.14, n=10), 2))

# question 12


def approx_cosh(x, n):
    cosh_value = 0
    for i in range(0, n):
        value = (np.power(x, (2*i)) / calc_factorial((2*i)))
        cosh_value += value
    return cosh_value


assert np.isclose(round(approx_cosh(x=1, n=10), 2),
                  1.54, rtol=1e-09, atol=1e-09)
print(round(approx_cosh(x=3.14, n=10), 2))
