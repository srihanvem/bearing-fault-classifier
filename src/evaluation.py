import numpy as np
import matplotlib.pyplot as plt


def compute_confusion_matrix(y_true, y_pred, num_classes: int):
    matrix = np.zeros((num_classes, num_classes), dtype=int)

    for actual, predicted in zip(y_true, y_pred):
        matrix[actual][predicted] += 1

    return matrix


def compute_metrics(confusion_matrix):
    num_classes = confusion_matrix.shape[0]
    total = np.sum(confusion_matrix)

    precision_scores = []
    recall_scores = []
    f1_scores = []

    for i in range(num_classes):
        true_positive = confusion_matrix[i, i]
        false_positive = np.sum(confusion_matrix[:, i]) - true_positive
        false_negative = np.sum(confusion_matrix[i, :]) - true_positive

        precision = true_positive / (true_positive + false_positive + 1e-10)
        recall = true_positive / (true_positive + false_negative + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)

        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

    accuracy = np.trace(confusion_matrix) / total

    return precision_scores, recall_scores, f1_scores, accuracy


def print_metrics(label_names, precision_scores, recall_scores, f1_scores, accuracy):
    print("\nModel Performance")
    print("-" * 60)
    print(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1-score':<12}")
    print("-" * 60)

    for name, precision, recall, f1 in zip(label_names, precision_scores, recall_scores, f1_scores):
        print(f"{name:<10} {precision:<12.2f} {recall:<12.2f} {f1:<12.2f}")

    print("-" * 60)
    print(f"Overall Accuracy: {accuracy:.2%}")


def plot_confusion_matrix(confusion_matrix, label_names, output_path="figures/confusion_matrix.png"):
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(confusion_matrix)

    ax.set_title("Confusion Matrix - KNN Bearing Fault Classifier")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    ax.set_xticks(np.arange(len(label_names)))
    ax.set_yticks(np.arange(len(label_names)))
    ax.set_xticklabels(label_names)
    ax.set_yticklabels(label_names)

    for i in range(len(label_names)):
        for j in range(len(label_names)):
            ax.text(j, i, confusion_matrix[i, j], ha="center", va="center")

    fig.colorbar(image)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()