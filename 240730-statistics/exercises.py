from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# question 1


def compute_mean(X):
    mean = np.mean(X)
    return mean


X = [2, 0, 2, 2, 7, 4, -2, 5, -1, -1]
print(" Mean : ", compute_mean(X))

# question 2


def compute_median(X):
    size = len(X)
    X = np.sort(X)
    print(X)
    if (size % 2) == 0:
        middle_position = int(size / 2 - 1)
        return (X[middle_position] + X[(middle_position + 1)]) * 0.5
    else:
        middle_position = int((size + 1) / 2 - 1)
        return X[middle_position]


X = [1, 5, 4, 4, 9, 13]
print(" Median : ", compute_median(X))

# question 3


def compute_std(X):
    mean = compute_mean(X)
    variance = np.sum((np.array(X) - mean) ** 2) * (1/len(X))
    return np.sqrt(variance)


X = [171, 176, 155, 167, 169, 182]
print(compute_std(X))

# question 4


def compute_correlation_coefficient(X, Y):
    N = len(X)
    numerator = 0
    denominator = 0

    numerator = N * np.sum(np.dot(X, Y)) - np.sum(X) * np.sum(Y)
    denominator = np.sqrt(N*np.sum(np.array(X)**2) - (np.sum(X))**2) * \
        np.sqrt(N*np.sum(np.array(Y)**2) - (np.sum(Y))**2)

    return np.round(numerator / denominator, 2)


X = np.asarray([-2, -5, -11, 6, 4, 15, 9])
Y = np.asarray([4, 25, 121, 36, 16, 225, 81])
print(" Correlation : ", compute_correlation_coefficient(X, Y))

# question 5


data = pd.read_csv('./data/advertising.csv')


def correlation(X, Y):
    X = np.asarray(X)
    Y = np.asarray(Y)
    return compute_correlation_coefficient(X, Y)


x = data['TV']
y = data['Radio']
corr_xy = correlation(x, y)
print(f" Correlation between TV and Sales : {round(corr_xy, 2)}")


# question 6

features = ['TV', 'Radio', 'Newspaper']
for feature_1 in features:
    for feature_2 in features:
        correlation_value = correlation(data[feature_1], data[feature_2])
        print(f" Correlation between {feature_1} and {feature_2}: {round(
            correlation_value, 2)}")

# question 7
x = data['Radio']
y = data['Newspaper']

result = np.corrcoef(x, y)
print(result)

# question 8
print(data.corr())

# question 9
plt.figure(figsize=(10, 8))
data_corr = data.corr()
sns.heatmap(data_corr, annot=True, fmt=".2f", linewidth=.5)
# plt.show()

# question 10
vi_data_df = pd.read_csv('./data/vi_text_retrieval.csv')
context = vi_data_df['text']
context = [doc.lower() for doc in context]

tfidf_vectorizer = TfidfVectorizer()
context_embedded = tfidf_vectorizer.fit_transform(context)
print(context_embedded.toarray()[7][0])

# question 10


def tfidf_search(question, tfidf_vectorizer, top_d=5):

    query_embedded = tfidf_vectorizer.transform([question.lower()])
    cosine_scores = cosine_similarity(
        context_embedded, query_embedded).reshape((-1,))

    results = []
    for idx in cosine_scores.argsort()[-top_d:][::-1]:
        doc_score = {
            'id': idx,
            'cosine_score': cosine_scores[idx]
        }
        results.append(doc_score)
    return results


question = vi_data_df.iloc[0]['question']
results = tfidf_search(question, tfidf_vectorizer, top_d=5)
print(results[0]['cosine_score'])

# question 12


def corr_search(question, tfidf_vectorizer, top_d=5):
    query_embedded = tfidf_vectorizer.transform([question.lower()])
    corr_scores = np.corrcoef(
        query_embedded.toarray()[0],
        context_embedded.toarray()
    )
    corr_scores = corr_scores[0][1:]
    results = []
    for idx in corr_scores.argsort()[-top_d:][::-1]:
        doc = {
            'id': idx,
            'corr_score': corr_scores[idx]
        }
        results.append(doc)
    return results


question = vi_data_df.iloc[0]['question']
results = corr_search(question, tfidf_vectorizer, top_d=5)
print(results[1]['corr_score'])
