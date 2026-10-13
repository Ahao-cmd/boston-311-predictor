# Boston 311 Predictor

A machine learning project that analyzes Boston's 311 service request data and predicts whether a service request is likely to be completed on time or overdue.

The project uses historical municipal service data, feature engineering, classification models, and threshold optimization to explore how machine learning can support early identification of potentially overdue requests.

## Overview

Boston 311 is a public service system that allows residents to report non-emergency issues such as street cleaning, parking enforcement, sanitation, and infrastructure maintenance.

This project uses Boston's 2024 service request records to build and evaluate machine learning models for predicting request outcomes.

The primary prediction task is binary classification:

- **ONTIME (0):** The request is recorded as completed on time.
- **OVERDUE (1):** The request is recorded as overdue.

The current version focuses on retrospective model evaluation rather than deployment in a live municipal service system.

## Key Features

- Official Boston 311 dataset collection
- Data cleaning and exploratory data analysis
- Time-based feature engineering
- Chronological train, validation, and test splits
- Logistic Regression classification
- Random Forest classification
- Classification threshold optimization
- Precision, Recall, F1, F2, and ROC-AUC evaluation
- Confusion matrix visualization
- Model comparison
- Trained model saving and loading using Joblib
- Single-request prediction using saved models

## Technologies Used

- Python
- pandas
- NumPy
- scikit-learn
- Matplotlib
- Requests
- Joblib
- Git and GitHub

## Dataset

**Source:** [Boston Open Data - 311 Service Requests](https://data.boston.gov/dataset/311-service-requests)

**Year:** 2024

**Total Records:** 306,756

The dataset includes information about service request categories, submission channels, neighborhoods, creation dates, case status, and service outcomes.

The raw dataset is excluded from this repository because of its file size. A download script is provided to retrieve the data from the official source.

## Machine Learning Workflow

The project follows these steps:

1. Download the official Boston 311 dataset.
2. Explore data distributions and missing values.
3. Clean and prepare historical service requests.
4. Engineer time-based features from request creation dates.
5. Split data chronologically to reduce temporal data leakage.
6. Train Logistic Regression and Random Forest models.
7. Select classification thresholds using validation data.
8. Evaluate models on a later test period.
9. Save trained models and generate predictions.
10. Visualize results and compare model performance.