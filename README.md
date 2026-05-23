# Bearing Fault Classifier

A machine learning project that detects and classifies ball bearing faults using vibration-derived statistical features.

## Project Overview

Ball bearings are used in many rotating machines, including motors, turbines, vehicles, and industrial equipment. When a bearing begins to fail, vibration patterns can reveal the problem before full failure occurs.

This project uses statistical features extracted from vibration signals to classify bearing condition into four categories:

- Ball fault
- Inner race fault
- Outer race fault
- Normal bearing

## Dataset

This project uses the Case Western Reserve University bearing dataset, accessed through Kaggle.

Dataset: https://www.kaggle.com/datasets/shayanfazeli/cwru

The dataset contains vibration signal features computed over 2048-point windows from a bearing test rig. Features include mean, standard deviation, RMS, skewness, kurtosis, crest factor, form factor, minimum, and maximum.

The dataset file is not included in this repository. To run the project, download the CSV file and place it in the `data/` folder as:

```text
data/feature_time_48k_2048_load_1.csv
```

## Method

The model uses a manually implemented k-Nearest Neighbors classifier. The workflow includes:

- Simplifying detailed fault labels into four main classes
- Standardizing the feature values
- Splitting the dataset into training and testing sets
- Predicting fault type using KNN
- Evaluating performance with precision, recall, F1-score, accuracy, and a confusion matrix

## Results

The KNN model achieved:

```text
Overall Accuracy: 94.57%
```

Class-level performance:

```text
Class      Precision    Recall       F1-score
ball       0.91         0.89         0.90
inner      0.99         1.00         0.99
outer      0.92         0.92         0.92
normal     0.98         0.98         0.98
```

The model performed especially well on inner race faults and normal bearing samples. Most errors occurred between ball faults and outer race faults, which suggests that some vibration patterns between those fault types are more similar.

## Project Structure

```text
src/
  data_utils.py      Data loading, label cleaning, and preprocessing
  knn_model.py       Manual KNN classifier
  evaluation.py      Metrics and confusion matrix plotting

data/                Local dataset folder, not uploaded to GitHub
figures/             Generated plots
outputs/             Model outputs
main.py              Main script
requirements.txt     Python dependencies
```

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the classifier:

```bash
python main.py
```

## Tools Used

- Python
- NumPy
- pandas
- matplotlib

## Future Improvements

Future versions could compare KNN against logistic regression, Naive Bayes, random forests, or neural networks. The project could also include feature importance analysis and testing across different bearing loads and fault sizes.