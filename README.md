# Predicting-Wine-Quality-using-Logistic-Regression
Project Overview

This project focuses on predicting the quality of wine using logistic regression, based on various physicochemical features of the wine dataset. The goal was to classify wines as either "good" (quality ≥ 6) or "not good" (quality < 6).

Through systematic data exploration, preprocessing, visualization, and model building, the analysis provides insights into the factors influencing wine quality and demonstrates how logistic regression can be applied for classification tasks.
Steps and Methodology

1. Libraries Used

NumPy: For numerical computations.
Pandas: For handling and preprocessing the dataset.
Matplotlib & Seaborn: For data visualization and exploratory analysis.
Scikit-learn: For splitting the dataset, training the logistic regression model, and evaluating performance.
Joblib: For saving the trained model.
Custom Module: OOPs_deep_stats (your own module, used as part of practice).

2. Data Preprocessing

Loaded the dataset WineQT.csv.
Inspected dataset shape, head, tail, info, and descriptive statistics.
Checked for null values (none found).
Dropped the unnecessary Id column.

3. Exploratory Data Analysis (EDA)

Plotted pairplots using seaborn to explore relationships between features and wine quality.
Compared variables such as acidity, residual sugar, chlorides, density, pH, sulphates, and alcohol against wine quality.
Observed how higher alcohol and sulphates tend to correlate with better wine quality.

4. Feature Selection and Target Variable

Features (X): All physicochemical properties except quality.
Target (y): Binary classification of wine quality (quality ≥ 6 → 1, else 0).

5. Model Building

Split the dataset into training (80%) and testing (20%), stratified by quality.
Applied Logistic Regression with increased iterations (max_iter=10000) to ensure convergence.
Trained the model on the training dataset.

6. Model Evaluation

Predictions were generated on the test set.
Evaluated using confusion matrix and classification report.
Reported metrics include precision, recall, F1-score, and accuracy.

7. Model Saving
Saved the final trained logistic regression model using joblib as
wine quality prediction model 76 f1 29-08-25.pkl.

Key Insights
Logistic regression was effective in distinguishing good-quality wines from lower-quality ones.

Alcohol content and sulphates were among the strongest indicators of wine quality.

The model achieved a reasonable balance between precision and recall, making it useful for classification tasks.
