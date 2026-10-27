# Customer Churn Prediction Using Machine Learning

## Overview

This project predicts customer churn using the Telco Customer Churn dataset. It applies machine learning classification techniques to identify customers who are likely to discontinue their telecom services.

Two models, Random Forest and XGBoost, are trained and evaluated. The project also addresses class imbalance, compares model performance using ROC-AUC, and identifies the most influential features affecting churn predictions.

## Objectives

- Explore and preprocess telecom customer data.
- Convert `TotalCharges` into numeric format and handle missing values.
- Encode categorical features using OneHotEncoder.
- Address class imbalance using class weighting.
- Train and compare Random Forest and XGBoost classifiers.
- Evaluate model performance using accuracy and ROC-AUC.
- Perform five-fold stratified cross-validation.
- Visualize ROC curves for both classifiers.
- Identify and visualize the top 10 important features for each model.

## Dataset

The project uses the Telco Customer Churn dataset, containing customer demographics, service subscriptions, contract details, and billing information.

- **Total records:** 7,043
- **Total columns:** 21
- **Target variable:** `Churn`

The target variable contains two classes:

- `Yes`: The customer discontinued the service.
- `No`: The customer continued the service.

### Features

The dataset includes features such as:

- Gender, Senior Citizen, Partner, and Dependents
- Tenure
- Phone and internet services
- Online Security and Tech Support
- Contract type
- Payment Method
- Monthly Charges
- Total Charges

The `customerID` column is removed before model training because it is an identifier rather than a useful customer behavior feature.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- XGBoost
- Jupyter Notebook

## Methodology

### 1. Data Exploration

Inspected the dataset's dimensions, data types, sample records, missing values, and churn class distribution.

The dataset contains 5,174 customers who did not churn and 1,869 customers who churned, indicating an imbalanced target distribution.

### 2. Data Preprocessing

- Converted blank entries in `TotalCharges` into missing values.
- Converted `TotalCharges` to numeric format.
- Imputed missing `TotalCharges` values using the median.
- Removed the `customerID` column.
- Split the data into training and testing sets using an 80/20 split with stratification.
- Applied OneHotEncoder to categorical features using a ColumnTransformer.

### 3. Handling Class Imbalance

Class weighting was used instead of SMOTE.

- **Random Forest:** Used `class_weight="balanced"`.
- **XGBoost:** Used `scale_pos_weight` based on the ratio of non-churn to churn cases in the training data.

These methods help the models account for the unequal distribution of target classes.

### 4. Model Training

Two classifiers were trained:

**Random Forest Classifier**

- 200 decision trees
- Balanced class weights
- Random state set to 42

**XGBoost Classifier**

- 200 boosting iterations
- Maximum tree depth of 5
- Learning rate of 0.05
- Class imbalance handled using `scale_pos_weight`
- Log-loss evaluation metric

### 5. Model Evaluation

The models were evaluated on the test set using:

- Accuracy
- ROC-AUC

ROC curves were plotted to compare the models' ability to distinguish between customers who churn and customers who remain.

### 6. Cross-Validation

Five-fold StratifiedKFold cross-validation was used to evaluate ROC-AUC on the training data.

Preprocessing and model training were combined in pipelines for cross-validation.

### 7. Feature Importance

Feature importance scores were extracted from both models. The top 10 features for each classifier were displayed in tables and visualized using horizontal bar charts.

## Results

### Test Set Performance

| Model         | Accuracy | ROC-AUC |
| ------------- | -------: | ------: |
| Random Forest |   78.42% |  82.20% |
| XGBoost       |   75.44% |  83.55% |

XGBoost achieved the higher test ROC-AUC score, while Random Forest achieved higher accuracy.

Both models exceeded the project's target ROC-AUC threshold of 0.80 on the test set.

### Five-Fold Cross-Validation Results

| Model         | Mean ROC-AUC | Standard Deviation |
| ------------- | -----------: | -----------------: |
| Random Forest |       0.8226 |             0.0117 |
| XGBoost       |       0.8451 |             0.0089 |

XGBoost achieved a higher mean cross-validation ROC-AUC than Random Forest.

### Most Important Features

**Random Forest's top features included:**

- Total Charges
- Tenure
- Monthly Charges
- Month-to-month Contract
- Online Security
- Two-year Contract
- Electronic Check Payment Method
- Tech Support
- Fiber Optic Internet Service
- Online Backup

**XGBoost's top features included:**

- Month-to-month Contract
- Fiber Optic Internet Service
- Online Security
- DSL Internet Service
- Streaming Movies
- One-year Contract
- Tech Support
- Two-year Contract
- Online Backup
- Electronic Check Payment Method

Feature importance indicates which features the models relied on most. It does not establish that these features cause customers to churn.

## Conclusion

This project demonstrates how machine learning can be used to predict customer churn from telecom subscription and billing information. Random Forest and XGBoost were trained with class imbalance handling, and their performance was evaluated using test-set metrics and five-fold cross-validation.

XGBoost achieved the best ROC-AUC results, reaching approximately 83.55% on the test set and 84.51% in cross-validation. Random Forest achieved higher test accuracy at approximately 78.42%.

The results show that both models exceeded the target test ROC-AUC of 80%, with XGBoost providing better discrimination between churn and non-churn customers in this experiment.
