import numpy as np


def main():

    greater_than_zero_note = 'n is not greater than 0'

    def calc_factorial(n):
        if (n == 0) or (n == 1):
            return 1
        else:
            return n * calc_factorial(n - 1)

    def approx_sin(x, n):
        if n > 0:
            sin_value = 0
            for i in range(0, n):
                value = np.power(-1, i) * (np.power(x, (2*i+1)
                                                    ) / calc_factorial((2*i+1)))
                sin_value += value
            return sin_value
        else:
            print(greater_than_zero_note)

    def approx_cos(x, n):
        if n > 0:
            cos_value = 0
            for i in range(0, n):
                value = np.power(-1, i) * \
                    (np.power(x, (2*i)) / calc_factorial((2*i)))
                cos_value += value
            return cos_value
        else:
            print(greater_than_zero_note)

    def approx_sinh(x, n):
        if n > 0:
            sinh_value = 0
            for i in range(0, n):
                value = (np.power(x, (2*i+1)) / calc_factorial((2*i+1)))
                sinh_value += value
            return sinh_value
        else:
            print(greater_than_zero_note)

    def approx_cosh(x, n):
        if n > 0:
            cosh_value = 0
            for i in range(0, n):
                value = (np.power(x, (2*i)) / calc_factorial((2*i)))
                cosh_value += value
            return cosh_value
        else:
            print(greater_than_zero_note)

    print(approx_sin(x=3.14, n=10))
    print(approx_cos(x=3.14, n=10))
    print(approx_sinh(x=3.14, n=10))
    print(approx_cosh(x=3.14, n=10))


if __name__ == '__main__':
    main()
