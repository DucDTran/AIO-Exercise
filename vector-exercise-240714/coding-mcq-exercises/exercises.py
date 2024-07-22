import cv2
import numpy as np
import os

# CODE
# question 1a


def compute_vector_length(vector):
    len_of_vector = np.linalg.norm(vector)
    return len_of_vector

# question 1b


def compute_dot_product(vector1, vector2):
    result = np.dot(vector1, vector2)
    return result

# question 1c


def matrix_multi_vector(matrix, vector):
    result = np.dot(matrix, vector)
    return result

# question 1d


def matrix_multi_matrix(matrix1, matrix2):
    result = np.dot(matrix1, matrix2)
    return result


# question 1e
def inverse_matrix(matrix):
    determinant = np.linalg.det(matrix)
    if determinant != 0:
        inverse = np.linalg.inv(matrix)
        return inverse
    else:
        return None

# question 2


def compute_eigenvalues_eigenvectors(matrix):
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    normalized_eigenvectors = eigenvectors / \
        np.linalg.norm(eigenvectors, axis=0)
    return eigenvalues, normalized_eigenvectors

# question 3


def compute_cosine(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    cosine_similarity = dot_product / (norm_v1 * norm_v2)
    return cosine_similarity


# question 4
bg1_image = cv2.imread('../images/GreenBackground.png', 1)
bg1_image = cv2.resize(bg1_image, (678, 381))


ob_image = cv2.imread('../images/Object.png', 1)
ob_image = cv2.resize(ob_image, (678, 381))


bg2_image = cv2.imread('../images/NewBackground.jpg', 1)
bg2_image = cv2.resize(bg2_image, (678, 381))


def compute_difference(bg_img, input_img):
    difference_three_channel = cv2.absdiff(bg_img, input_img)
    difference_single_channel = np.sum(difference_three_channel, axis=2) / 3.0
    difference_single_channel = difference_single_channel.astype('uint8')
    return difference_single_channel


difference_single_channel = compute_difference(bg1_image, ob_image)
# cv2.imshow('Difference', difference_single_channel)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


def compute_binary_mask(difference_single_channel):

    difference_binary = np.where(difference_single_channel >= 15, 255, 0)
    difference_binary = np.stack((difference_binary, )*3, axis=-1)
    return difference_binary


binary_mask = compute_binary_mask(difference_single_channel)
# cv2.imshow('Binary Mask', binary_mask)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


def replace_background(bg1_image, bg2_image, ob_image):
    difference_single_channel = compute_difference(
        bg1_image,
        ob_image)

    binary_mask = compute_binary_mask(difference_single_channel)

    output = np.where(binary_mask == 255, ob_image, bg2_image)

    return output


output = replace_background('./GreenBackground.png',
                            './NewBackground.jpg', './Object.png')
cv2.imshow('Output', output)
cv2.waitKey(0)
cv2.destroyAllWindows()


# MCQ
# question 1
vector = np.array([-2, 4, 9, 21])
result = compute_vector_length([vector])
print(round(result, 2))

# question 2
v1 = np.array([0, 1, -1, 2])
v2 = np.array([2, 5, 1, 0])
result = compute_dot_product(v1, v2)
print(round(result, 2))

# question 3
x = np.array([[1, 2], [3, 4]])
k = np.array([1, 2])
print('result \n', x.dot(k))

# question 4
x = np.array([[-1, 2], [3, -4]])
k = np.array([1, 2])
print('result \n', x@k)


# question 5
m = np.array([[-1, 1, 1], [0, -4, 9]])
v = np.array([0, 2, 1])
result = matrix_multi_vector(m, v)
print(result)

# question 6
m1 = np.array([[0, 1, 2], [2, -3, 1]])
m2 = np.array([[1, -3], [6, 1], [0, -1]])
result = matrix_multi_matrix(m1, m2)
print(result)

# question 7
m1 = np.eye(3)
m2 = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
result = m1@m2
print(result)


# question 8
m1 = np.eye(2)
m1 = np.reshape(m1, (-1, 4))[0]
m2 = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]])
result = m1@m2
print(result)

# question 9
m1 = np.array([[1, 2], [3, 4]])
m1 = np.reshape(m1, (-1, 4), "F")[0]
m2 = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]])
result = m1@m2
print(result)

# question 10
m1 = np.array([[-2, 6], [8, -4]])
result = inverse_matrix(m1)
print(result)

# question 11
matrix = np.array([[0.9, 0.2], [0.1, 0.8]])
eigenvalues, eigenvectors = compute_eigenvalues_eigenvectors(matrix)
print(eigenvectors)


# question 12
x = np.array([1, 2, 3, 4])
y = np.array([1, 0, 3, 0])
result = compute_cosine(x, y)
print(round(result, 3))
