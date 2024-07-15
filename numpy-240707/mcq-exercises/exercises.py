
import pandas as pd
import matplotlib.image as mpimg
import numpy as np

# question 12
img = mpimg.imread('C:\\Users\\dinhd\\AIO-Exercise\\numpy-240707\\dog.jpeg')


def convert_to_grayscale_lightness(image):
    if len(image.shape) == 3 and image.shape[2] == 3:
        lightness = (np.max(image, axis=2) + np.min(image, axis=2)) / 2
        return lightness
    else:
        return image


gray_image_01 = convert_to_grayscale_lightness(img)
print(gray_image_01[0, 0])

# question 13


def convert_to_grayscale_average(image):
    if len(image.shape) == 3 and image.shape[2] == 3:
        average = np.mean(image, axis=2)
        return average
    else:
        return image


gray_image_02 = convert_to_grayscale_average(img)
print(gray_image_02[0, 0])

# question 14


def convert_to_grayscale_luminosity(image):
    if len(image.shape) == 3 and image.shape[2] == 3:
        R = image[:, :, 0]
        G = image[:, :, 1]
        B = image[:, :, 2]

        # Compute the luminosity
        luminosity = 0.21 * R + 0.72 * G + 0.07 * B
        return luminosity
    else:
        return image


gray_image_03 = convert_to_grayscale_luminosity(img)
print(gray_image_03[0, 0])

# question 15
df = pd.read_csv(
    'C:\\Users\\dinhd\\AIO-Exercise\\numpy-240707\\advertising.csv')
data = df.to_numpy()
print(df.columns)
min_sales = np.min(data[:, 3])
max_sales = np.max(data[:, 3])
max_sales_idx = np.argmax(data[:, 3])
print(min_sales, max_sales, max_sales_idx)

# question 16
avg_tv = np.mean(data[:, 0])
print(avg_tv)

# question 17
print(len(data[data[:, 3] >= 20]))

# question 18
print(np.mean(data[data[:, 3] >= 15][:, 1]))

# question 19
print(np.sum(data[data[:, 2] > np.mean(data[:, 2])][:, 3]))

# question 20


def my_func(x, average_value):
    if x > average_value:
        return "Good"
    elif x < average_value:
        return "Bad"
    else:
        return "Average"


A = np.mean(data[:, 3])
vectorized_func = np.vectorize(lambda x: my_func(x, A))
scores = vectorized_func(data[:, 3])
print(scores[7:10])


# question 21
sales_data = data[:, 3]
mean_sales = np.mean(sales_data)
A = sales_data[np.abs(sales_data - mean_sales).argmin()]
vectorized_func = np.vectorize(lambda x: my_func(x, A))
scores = vectorized_func(data[:, 3])
print(scores[7:10])
