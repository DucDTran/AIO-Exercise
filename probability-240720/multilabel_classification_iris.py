import numpy as np
import math


def create_train_data_iris():
    with open('./data/iris.data.txt') as f:
        data = []
        for line in f:
            data.append(line.strip().split(','))
        data_array = np.array(data)
    return data_array


def compute_prior_probability(train_data):
    y_unique = np.unique(train_data[:, 4])
    prior_probability = np.zeros(len(y_unique))
    total_samples = train_data.shape[0]
    for i, y in enumerate(y_unique):
        prior_probability[i] = np.count_nonzero(
            train_data[:, 4] == y) / total_samples
    return prior_probability


def compute_conditional_probability(train_data):
    y_unique = np.unique(train_data[:, 4])
    conditional_probability = []
    for i in range(0, train_data.shape[1]-1):
        x_conditional_probability = np.zeros((len(y_unique), 2))
        for j in range(0, len(y_unique)):
            feature_data = train_data[:, i][train_data[:, 4] == y_unique[j]].astype(
                float)
            mean = np.mean(feature_data)
            sigma = np.std(feature_data)
            sigma = sigma * sigma  # variance
            x_conditional_probability[j] = [mean, sigma]
        conditional_probability.append(x_conditional_probability)
    return conditional_probability


def compute_gaussian_distribution(number, mean, sigma):
    result = (1.0 / (np.sqrt(2 * math.pi * sigma))) * \
        (np.exp(-(float(number) - mean) ** 2 / (2 * sigma)))
    return result


def train_gaussian_naive_bayes(train_data):
    prior_probability = compute_prior_probability(train_data)
    conditional_probability = compute_conditional_probability(train_data)
    return prior_probability, conditional_probability


def prediction_iris(X, prior_probability, conditional_probability):
    probabilities = []
    for class_idx in range(len(prior_probability)):
        prob = prior_probability[class_idx]
        for feature_idx in range(len(X)):
            mean = conditional_probability[feature_idx][class_idx][0]
            sigma = conditional_probability[feature_idx][class_idx][1]
            prob *= compute_gaussian_distribution(X[feature_idx], mean, sigma)
        probabilities.append(prob)
    return np.argmax(probabilities)


# Example 1
# X = [sepal length, sepal width, petal length, petal width]
X = [6.3, 3.3, 6.0, 2.5]
train_data = create_train_data_iris()
y_unique = np.unique(train_data[:, 4])
prior_probability, conditional_probability = train_gaussian_naive_bayes(
    train_data)
pred = y_unique[prediction_iris(X, prior_probability, conditional_probability)]
print(pred)
assert pred == "Iris-virginica"

# Example 2 #########################
# X = [sepal length, sepal width, petal length, petal width]
X = [5.0, 2.0, 3.5, 1.0]
prior_probability, conditional_probability = train_gaussian_naive_bayes(
    train_data)
pred = y_unique[prediction_iris(X, prior_probability, conditional_probability)]
print(pred)
assert pred == "Iris-versicolor"

# Example 3 #########################
# X = [sepal length, sepal width, petal length, petal width]
X = [4.9, 3.1, 1.5, 0.1]
prior_probability, conditional_probability = train_gaussian_naive_bayes(
    train_data)
pred = y_unique[prediction_iris(X, prior_probability, conditional_probability)]
print(pred)
assert pred == "Iris-setosa"
