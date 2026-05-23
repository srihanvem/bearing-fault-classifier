from src.data_utils import load_and_prepare_data
from src.knn_model import predict_knn
from src.evaluation import (
    compute_confusion_matrix,
    compute_metrics,
    print_metrics,
    plot_confusion_matrix,
)


DATA_PATH = "data/feature_time_48k_2048_load_1.csv"
K_VALUE = 5


def main():
    X_train, X_test, y_train, y_test, label_names = load_and_prepare_data(DATA_PATH)

    y_pred = predict_knn(X_train, y_train, X_test, k=K_VALUE)

    confusion_matrix = compute_confusion_matrix(
        y_true=y_test,
        y_pred=y_pred,
        num_classes=len(label_names),
    )

    precision, recall, f1, accuracy = compute_metrics(confusion_matrix)

    print_metrics(label_names, precision, recall, f1, accuracy)
    plot_confusion_matrix(confusion_matrix, label_names)


if __name__ == "__main__":
    main()