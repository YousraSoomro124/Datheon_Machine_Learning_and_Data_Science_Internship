# Sentiment Analysis Using NLP and Machine Learning

## Project Overview

This project uses Natural Language Processing (NLP) and machine learning to classify movie reviews as **positive or negative**. It applies text preprocessing and TF-IDF vectorization, then trains and evaluates two classification models: Multinomial Naive Bayes and Logistic Regression.

## Dataset

- **Dataset:** [IMDb Movie Reviews Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- **Original size:** 50,000 reviews
- **Columns:** `review`, `sentiment`
- **Classes:** Positive and Negative
- **After removing duplicates:** 49,582 reviews

The dataset contains balanced positive and negative movie reviews.

## Technologies and Libraries

- Python
- Pandas
- NLTK
- Regular Expressions (`re`)
- Scikit-learn

## Project Workflow

### 1. Data Loading and Exploration

Loaded the IMDb dataset using Pandas and examined its structure, missing values, duplicate records, and sentiment distribution.

### 2. Data Cleaning and Preprocessing

Applied the following text preprocessing techniques:

- Converted text to lowercase.
- Removed HTML tags.
- Removed punctuation and non-alphabetic characters.
- Removed common English stopwords
- Applied lemmatization using NLTK's WordNetLemmatizer.

Duplicate records were removed before model training.

### 3. Train-Test Split

Split the dataset into training and testing sets using an 80:20 ratio.

- **Training samples:** 39,665
- **Testing samples:** 9,917
- **Random state:** 42
- **Stratification:** Applied to maintain similar sentiment proportions in both sets.

### 4. TF-IDF Vectorization

Converted cleaned reviews into numerical features using `TfidfVectorizer`.

- Maximum features: 5,000
- Training matrix shape: `(39665, 5000)`
- Testing matrix shape: `(9917, 5000)`

The vectorizer was fitted on the training data and then used to transform the testing data.

### 5. Model Training

Trained and compared two machine learning classifiers:

- **Multinomial Naive Bayes**
- **Logistic Regression**

### 6. Model Evaluation

Evaluated both models on the held-out testing dataset using accuracy, precision, recall, and F1-score.

| Metric          | Multinomial Naive Bayes | Logistic Regression |
| --------------- | ----------------------: | ------------------: |
| Accuracy        |                  85.47% |              88.94% |
| Macro Precision |                    0.85 |                0.89 |
| Macro Recall    |                    0.85 |                0.89 |
| Macro F1-score  |                    0.85 |                0.89 |

**Result:** Logistic Regression achieved the higher test accuracy of **88.94%**, outperforming Multinomial Naive Bayes on this dataset and configuration.

### 7. Custom Sentiment Prediction

Implemented a `predict_sentiment()` function that accepts a new movie review, preprocesses it, transforms it using the fitted TF-IDF vectorizer, and predicts its sentiment using the trained Logistic Regression model.

Example:

```python
print(predict_sentiment("This movie was amazing and wonderful!"))
# Output: positive

print(predict_sentiment("This movie was boring and terrible."))
# Output: negative
```

## Key Learning Outcomes

- Text preprocessing for NLP tasks
- TF-IDF feature extraction
- Training and comparing text classification models
- Evaluating classifiers using multiple metrics
- Building a reusable function for custom sentiment predictions

## Conclusion

This project demonstrates an end-to-end sentiment classification workflow using NLP and traditional machine learning. Logistic Regression achieved **88.94% accuracy** on the test set, making it the better-performing model among the two evaluated in this experiment.
