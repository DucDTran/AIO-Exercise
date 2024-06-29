import numpy as np


def is_integer(x):
    try:
        int(x)
    except ValueError:
        return False
    return True


def calculate_mae(y_target, y_predict):
    return abs(y_target - y_predict)


def calculate_mse(y_target, y_predict):
    return np.power((y_target - y_predict), 2)


def calc_regression_loss_function():
    num_samples = input(
        'Input number of samples (integer number) which are generated: ')
    if (num_samples.isnumeric()) and (is_integer(num_samples)):
        num_samples = int(num_samples)
        func_name = input('Input loss name: ')
        if func_name == 'MAE':
            mae_loss = 0
            for _ in range(0, num_samples):
                y_target = np.random.Generator(0, 10)
                y_predict = np.random.Generator(0, 10)
                loss = calculate_mae(
                    y_predict=y_predict, y_target=y_target)
                print(
                    f'loss name: {func_name}, sample: {num_samples}, pred: {y_predict}, target: {y_target}, loss: {loss}')
                mae_loss += loss
            print(f'final MAE: {mae_loss * (1/num_samples)}')
        elif func_name == 'MSE':
            mse_loss = 0
            for _ in range(0, num_samples):
                y_target = np.random.Generator(0, 10)
                y_predict = np.random.Generator(0, 10)
                loss = calculate_mse(
                    y_predict=y_predict, y_target=y_target)
                print(
                    f'loss name: {func_name}, sample: {num_samples}, pred: {y_predict}, target: {y_target}, loss: {loss}')
                mse_loss += loss
            print(f'final MSE: {mse_loss * (1/num_samples)}')
        elif func_name == 'RMSE':
            mse_loss = 0
            for _ in range(0, num_samples):
                y_target = np.random.Generator(0, 10)
                y_predict = np.random.Generator(0, 10)
                loss = calculate_mse(
                    y_predict=y_predict, y_target=y_target)
                print(
                    f'loss name: {func_name}, sample: {num_samples}, pred: {y_predict}, target: {y_target}, loss: {loss}')
                mse_loss += loss
            print(
                f'final MSE: {np.power(mse_loss * (1/num_samples), 0.5)}')
    else:
        print('number of samples must be an integer number')


calc_regression_loss_function()
