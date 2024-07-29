import numpy as np
import os


def create_train_data():

    with open('./data/playtennis.txt') as f:
        data = []
        for line in f:
            data.append(line.strip().split(' '))
        data_array = np.array(data)
        data_array = data_array[1:, 1:]

    return data_array


def compute_prior_probability(train_data):
    y_unique = ['No', 'Yes']
    prior_probability = np.zeros(len(y_unique))
    total_samples = train_data.shape[0]
    for i, y in enumerate(y_unique):
        prior_probability[i] = np.count_nonzero(
            train_data[:, -1] == y) / total_samples
    return prior_probability


def compute_conditional_probability(train_data):
    y_unique = ['No', 'Yes']
    conditional_probability = []
    list_x_name = []
    for i in range(0, train_data.shape[1]-1):
        x_unique = np.unique(train_data[:, i])
        print('x_unique:', x_unique)
        list_x_name.append(x_unique)

        conditional_prob = np.zeros((len(y_unique), len(x_unique)))
        for y_idx, y_value in enumerate(y_unique):
            for x_idx, x_value in enumerate(x_unique):
                count_xy = np.count_nonzero(
                    (train_data[:, i] == x_value) & (train_data[:, -1] == y_value))
                count_y = np.count_nonzero(train_data[:, -1] == y_value)
                conditional_prob[y_idx, x_idx] = count_xy / \
                    count_y if count_y != 0 else 0
        conditional_probability.append(conditional_prob)

    return conditional_probability, list_x_name


def get_index_from_value(feature_name, list_features):
    index = np.nonzero(list_features == feature_name)
    if index[0].size > 0:
        return index[0][0]
    else:
        raise ValueError(f"Feature name '{
                         feature_name}' not found in the list of features.")


def train_naive_bayes(train_data):

    prior_probability = compute_prior_probability(train_data=train_data)

    conditional_probability, list_x_name = compute_conditional_probability(
        train_data=train_data)

    return prior_probability, conditional_probability, list_x_name


def prediction_play_tennis(X, list_x_name, prior_probability, conditional_probability):

    x1 = get_index_from_value(X[0], list_x_name[0])
    x2 = get_index_from_value(X[1], list_x_name[1])
    x3 = get_index_from_value(X[2], list_x_name[2])
    x4 = get_index_from_value(X[3], list_x_name[3])

    p0 = 0
    p1 = 0

    p_no = prior_probability[0]
    p_yes = prior_probability[1]

    p_x_no = 1
    p_x_yes = 1

    for i, x in enumerate([x1, x2, x3, x4]):
        p_x_yes *= np.round(conditional_probability[i][1, x], 2)
        p_x_no *= np.round(conditional_probability[i][0, x], 2)

    p0_unnormalized = p_no * p_x_no
    p1_unnormalized = p_yes * p_x_yes

    print("P(Play Tennis '= 'Yes' | X) = ", p1_unnormalized)
    print("P(Play Tennis '= 'No' | X) = ", p0_unnormalized)

    p0 = p0_unnormalized / (p0_unnormalized + p1_unnormalized)
    p1 = p1_unnormalized / (p0_unnormalized + p1_unnormalized)

    if p0 > p1:
        y_pred = 0
    else:
        y_pred = 1

    return y_pred


# question 14

train_data = create_train_data()
print(train_data)
prior_probability = compute_prior_probability(train_data=train_data)
print(prior_probability)
conditional_probability, list_x_name = compute_conditional_probability(
    train_data=train_data)
print(len(conditional_probability))

# question 15

print("x1 = ", list_x_name[0])
print("x2 = ", list_x_name[1])
print("x3 = ", list_x_name[2])
print("x4 = ", list_x_name[3])

# question 16

outlook = list_x_name[0]
i1 = get_index_from_value("Overcast", outlook)
i2 = get_index_from_value("Rain", outlook)
i3 = get_index_from_value("Sunny", outlook)
print(i1, i2, i3)

# question 17

x1 = get_index_from_value('Sunny', list_x_name[0])
print("P('Outlook'= 'Sunny' | Play Tennis '= 'Yes') = ",
      np.round(conditional_probability[0][1, x1], 2))

# question 18

print("P('Outlook'= 'Sunny' | Play Tennis '= 'No') = ",
      np.round(conditional_probability[0][0, x1], 2))

# question 19

X = ['Sunny', 'Cool', 'High', 'Strong']
data = create_train_data()
prior_probability, conditional_probability, list_x_name = train_naive_bayes(
    data)
pred = prediction_play_tennis(X, list_x_name, prior_probability,
                              conditional_probability)
if (pred):
    print("Ad should go!")
else:
    print("Ad should not go!")
