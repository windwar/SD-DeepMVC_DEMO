import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score, adjusted_rand_score, fowlkes_mallows_score, v_measure_score


def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    size = max(y_pred.max(), y_true.max()) + 1
    counts = np.zeros((size, size), dtype=np.int64)
    for index in range(y_pred.size):
        counts[y_pred[index], y_true[index]] += 1
    rows, columns = linear_sum_assignment(counts.max() - counts)
    return counts[rows, columns].sum() / y_pred.size


def purity(y_true, y_pred):
    truth = y_true.copy()
    labels = np.unique(truth)
    for index, label in enumerate(labels):
        truth[truth == label] = index
    bins = np.append(np.unique(truth), np.max(truth) + 1)
    voted = np.zeros(truth.shape)
    for cluster in np.unique(y_pred):
        histogram, _ = np.histogram(truth[y_pred == cluster], bins=bins)
        voted[y_pred == cluster] = np.argmax(histogram)
    return accuracy_score(truth, voted)


def evaluate(label, pred):
    return (
        v_measure_score(label, pred),
        adjusted_rand_score(label, pred),
        cluster_acc(label, pred),
        purity(label, pred),
        fowlkes_mallows_score(label, pred),
    )
