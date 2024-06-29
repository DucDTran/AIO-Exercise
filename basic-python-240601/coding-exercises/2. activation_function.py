import math
import numpy as np


def main():

    def is_number(x):
        try:
            float(x)
        except ValueError:
            return False
        return True

    def sigmoid_func(x):
        return 1 / (1 + np.exp(-x))

    def relu_func(x):
        return 0 if x <= 0 else x

    def elu_func(x):
        return 0.01 * (math.exp(x) - 1) if x <= 0 else x

    def activation_function():
        x = input('Input x = ')
        if is_number(x):
            x = float(x)
            func_name = str(
                input('Input activation function (sigmoid|relu|elu): '))
            if func_name == 'sigmoid':
                print(f'sigmoid: f({x}) = ', sigmoid_func(x))
            elif func_name == 'relu':
                print(f'relu: f({x}) = ', relu_func(x))
            elif func_name == 'elu':
                print(f'elu: f({x}) = ', elu_func(x))
            else:
                print(f'{func_name} is not supported')
        else:
            print('x must be a number')
    activation_function()


if __name__ == '__main__':
    main()
