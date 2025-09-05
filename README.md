# Intro to Machine Learning with TensorFLow Nano Degree: Finding Donors for CharityML

Welcome to the **first project** of the Data Scientist Nanodegree! In this project, the goal is to predict whether an individual earns more than $50,000 per year using data collected from the **1994 U.S. Census**. This information can help CharityML, a non-profit organization, identify potential donors more accurately.  

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Data Exploration](#data-exploration)
- [Data Preprocessing](#data-preprocessing)
- [Supervised Learning Models](#supervised-learning-models)
- [Model Evaluation](#model-evaluation)
- [Model Tuning](#model-tuning)
- [Feature Importance](#feature-importance)
- [Results](#results)
- [References](#references)

---

## Project Overview
The goal of this project is to:
1. Explore the census dataset and understand its features.
2. Preprocess the data by transforming skewed features, normalizing numerical features, and encoding categorical variables.
3. Apply supervised learning algorithms to predict whether an individual earns more than $50,000 per year.
4. Evaluate and tune the models for optimal performance.
5. Determine the most important features for predicting income.

---

## Dataset
The dataset originates from the **UCI Machine Learning Repository**, collected by Ron Kohavi and Barry Becker. The dataset has been cleaned, removing invalid or missing entries and excluding the `fnlwgt` feature.  

**Target variable:** `income` (binary: `<=50K` or `>50K`)  
**Number of records:** 45,222  

**Features:**
- Continuous: `age`, `education-num`, `capital-gain`, `capital-loss`, `hours-per-week`
- Categorical: `workclass`, `education_level`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `native-country`

---

## Data Exploration
- Individuals earning more than $50,000: **11,208**  
- Individuals earning at most $50,000: **34,014**  
- Percentage of individuals earning more than $50,000: **24.78%**

This indicates a class imbalance, which is important when evaluating model performance.

---

## Data Preprocessing
1. **Log-transform skewed features**: `capital-gain` and `capital-loss`  
2. **Normalization**: Using `MinMaxScaler` for continuous features.  
3. **One-hot encoding**: All categorical features were converted into numerical dummy variables.  
4. **Target encoding**: `income` was mapped to 0 (`<=50K`) and 1 (`>50K`).

**Number of features after encoding:** 103  

---

## Supervised Learning Models
Three supervised learning models were chosen to predict income:

### 1. Decision Tree Classifier
- **Application:** Credit scoring in finance  
- **Strengths:** Intuitive, handles numerical and categorical data, captures non-linear relationships  
- **Weaknesses:** Prone to overfitting, sensitive to small dataset changes  
- **Suitability:** Handles categorical census features directly and is easy to visualize

### 2. Random Forest Classifier
- **Application:** Fraud detection in e-commerce  
- **Strengths:** Reduces overfitting, robust to noise, handles large datasets well  
- **Weaknesses:** Less interpretable than single trees, slower training on very large datasets  
- **Suitability:** Best for mixed feature types and provides high accuracy for the census dataset

### 3. Logistic Regression
- **Application:** Predicting disease occurrence in healthcare  
- **Strengths:** Simple, interpretable, outputs probabilities  
- **Weaknesses:** Assumes linear relationships, requires proper encoding  
- **Suitability:** Binary target fits perfectly, can scale well to large datasets

---

## Model Evaluation
A **naive predictor** that predicts all individuals earn more than $50,000 yields:
- **Accuracy:** 0.2478  
- **F-score:** 0.2917  

The supervised learning models were trained on 1%, 10%, and 100% of the training data, evaluated with accuracy and F0.5 score.  

**Result highlights:**
- **Random Forest Classifier** achieved the highest F-score on the test set (~0.70) and maintained fast prediction times.
- **Decision Tree** performed reasonably but overfit on smaller datasets.
- **Logistic Regression** was fast but slightly lower in F-score.

---

## Model Tuning
The Random Forest Classifier was further optimized using **GridSearchCV**:
- Parameters tuned: `n_estimators`, `max_depth`, `min_samples_split`
- **Unoptimized Model Performance:**  
  - Accuracy: 0.8405  
  - F-score: 0.6769  
- **Optimized Model Performance:**  
  - Accuracy: 0.8541  
  - F-score: 0.7240  

This demonstrates an improvement over the unoptimized model and significant improvement over the naive predictor.

---

## Feature Importance
The five most important features for predicting income were:
1. `education-num`  
2. `marital-status`  
3. `capital-gain`  
4. `hours-per-week`  
5. `age`  

These align with expectations, highlighting the impact of education, household situation, additional income sources, work hours, and experience.

---

## Results
| Metric             | Naive Predictor | Unoptimized Model | Optimized Model |
|-------------------|----------------|-----------------|----------------|
| Accuracy           | 0.2478         | 0.8405          | 0.8541         |
| F-score            | 0.2917         | 0.6769          | 0.7240         |



