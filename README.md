# Yelp Rating Prediction using XGBoost and Spark

## Overview
This project implements a **Yelp rating prediction system** as part of a data mining competition. The objective is to predict star ratings for given user–business pairs while minimizing **Root Mean Squared Error (RMSE)**. The final solution improves upon a baseline recommendation system by using a **feature-rich supervised learning approach** combined with **Spark RDD-based data processing** for scalability.

---

## Problem Statement
Given historical Yelp data, predict the rating a user would assign to a business. The evaluation metric is RMSE, with stricter penalties for larger prediction errors.

---

## Approach
- Replaced traditional collaborative filtering with a **supervised regression model (XGBoost)** to improve prediction accuracy and efficiency.
- Used **Apache Spark RDDs exclusively** for data preprocessing to comply with competition constraints.
- Engineered detailed **user-level and business-level features** to capture behavioral patterns, engagement, and business quality.
- Performed manual hyperparameter tuning using validation RMSE.

---

## Feature Engineering
A total of **27 features** were extracted:

### Business Features (10)
- Average stars  
- Review count  
- Latitude & longitude  
- Price range  
- Credit card acceptance  
- Appointment-only availability  
- Reservations  
- Table service  
- Wheelchair accessibility  

### User Features (17)
- Review count  
- Friends count  
- Useful, funny, cool votes  
- Fans count  
- Elite years  
- Average stars  
- Compliments (hot, profile, list, note, plain, cool, funny, writer, photos)

Missing values were handled using **default values or randomized imputation** to ensure model robustness.

---

## Model Details
- **Model:** XGBoost Regressor  
- **Hyperparameters:**
  - Number of trees: 1000  
  - Max depth: 5  
  - Learning rate (eta): 0.05  
  - Subsample: 0.9  
  - Column sample by tree: 0.7  
  - Regularization: L1 = 0.3, L2 = 1.0  

The model was trained on Yelp training data and evaluated on a validation dataset.

---

## Results
- **Validation RMSE:** `0.9791`  
- **Error Distribution:**
  - ≥0 and <1: 102,277  
  - ≥1 and <2: 32,735  
  - ≥2 and <3: 6,198  
  - ≥3 and <4: 833  
  - ≥4: 1  
- **Execution Time:** ~367 seconds  

The model outperformed the baseline recommendation system and met the competition performance threshold.

---

## How to Run

### Requirements
- Python 3.6  
- Apache Spark 3.1.2  
- XGBoost  
- NumPy  

### Command
```bash
spark-submit competition.py <data_folder> <test_file_path> <output_file_path>
