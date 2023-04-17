# SVM Classification - Performs the cross validation and outputs the evaluation metrics.
import pandas as pd
from classification.SVM import optimize_hyperparams, classify, print_confusion_matrix, find_misclassified

# Read features and labels from CSV
df = pd.read_csv(filepath_or_buffer='../data/output/features/NC16_extracted_features.csv')

# Separate features and labels
X = df.loc[:, ~df.columns.isin(['labels', 'image_names'])]
y = df['labels']

# Keep track of image ids for later use
img_ids = df['image_names']

# Check if there are any NaN values in the dataset
print('Has NaN:', df.isnull().values.any())

# Define hyperparameters for the SVM
hyper_params = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}]

# Optimize hyperparameters using GridSearchCV
opt_params = optimize_hyperparams(X, y, params=hyper_params)

# Train the SVM classifier and print evaluation metrics
classify(X, y, opt_params)
print_confusion_matrix(X, y, opt_params)
find_misclassified(X, y, opt_params, img_ids)
