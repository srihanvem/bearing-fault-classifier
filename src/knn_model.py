import numpy as np


def predict_knn(X_train, y_train, X_test, k: int = 5):
    predictions = []

    for x in X_test:
        distances = np.sqrt(np.sum((X_train - x) ** 2, axis=1))
        nearest_indices = np.argsort(distances)[:k]
        nearest_labels = y_train[nearest_indices]

        labels, counts = np.unique(nearest_labels, return_counts=True)
        predicted_label = labels[np.argmax(counts)]
        predictions.append(predicted_label)

    return np.array(predictions)