# Regression Project and R-Programming Report
![Last Commit](https://img.shields.io/github/last-commit/JPP-J/regression_project?style=flat-square)
![Python](https://img.shields.io/badge/Python-100%25-blue?style=flat-square)
![Languages](https://img.shields.io/github/languages/count/JPP-J/regression_project?style=flat-square)

This repository contains the code and report accompanying Jidapa's *Regression Project and R-Programming Report*, featuring:

## 📌 Overview

This project analyzes Amazon sales data to build a **multiple linear regression model** that predicts total revenue based on key product attributes. It is complemented by an R programming report focusing on statistical hypothesis testing techniques.

### 🧩 Problem Statement

Accurately forecasting total revenue from product data is vital for e-commerce platforms like Amazon. The aim of this project is to develop a regression model that uses **unit price** and **unit cost** to predict total revenue, helping businesses make informed pricing and cost-control decisions.

### 🔍 Approach

A multiple linear regression model was developed using Python. The workflow included feature selection through correlation analysis, data preprocessing, and evaluation using key regression metrics. In parallel, hypothesis testing was conducted in R to support statistical understanding of the data.

### 🎢 Processes

1. **Data Preprocessing** – Handle missing values, normalize scales, and prepare input features (`unit_price`, `unit_cost`)  
2. **Feature Selection** – Apply correlation matrix and domain knowledge to retain impactful predictors  
3. **Exploratory Data Analysis (EDA)** – Visualize feature relationships with revenue  
4. **Model Training** – Train regression model on a 90/10 train-test split  
5. **Model Evaluation** – Evaluate using R-squared and Mean Squared Error (MSE)  
6. **Hypothesis Testing (R)** – Perform Chi-Square Test, Logit Model example test
### 🎯 Results & Impact

- **R-squared:** 0.7451 — model explains ~74.51% of total revenue variance  
- **MSE:** ~381 billion — signals room for reducing prediction error  
- **Insights:** Unit price and unit cost are confirmed as significant predictors  

### ⚙️ Model Development Challenges

- **Limited Data Size:** Small sample (~100 entries) increases the risk of overfitting  
- **Prediction Error:** Despite good R-squared, high MSE suggests model could benefit from additional features  
- **Linear Assumptions:** Assumes a linear relationship between inputs and output, which might not capture complexity in real-world pricing behavior  
- **Feature Limitations:** Reliance on just two predictors limits model’s ability to generalize

## Conclusion

The linear regression model showed strong predictive ability for total revenue from Amazon sales data. Unit price and unit cost were confirmed as significant predictors, supported by correlation analysis, contributing to the model’s power.

- The model explained approximately **74.51%** of the variance in total revenue (R-squared = 0.7451).
- While achieving satisfactory accuracy, the Mean Squared Error (MSE) of around **381 billion** indicates room for improvement in reducing prediction error.
- The model provides valuable insights into how pricing and cost management affect revenue, which can help optimize pricing strategies and cost control to maximize revenue.

## Suggestions for Future Work

- The current sample size (100) may be too small, increasing overfitting risk; consider collecting more data to improve model robustness.
- Experiment with alternative models, such as decision trees or models capturing non-linear relationships, to enhance predictive performance.
- Apply further feature engineering to extract more informative predictors.


## Usage

- Run the regression analysis in Python with [`main.py`](main.py).
- Refer to the comprehensive report for hypothesis testing and statistical details in [`R_hypothesis_testing_report.pdf`](R_hypothesis_testing_report.pdf).


## References

- [Kaggle Linear Regression Code](https://kaggle.com/code/jidapapooljan/linear-regression)


*This project provides a solid baseline for revenue prediction and demonstrates how combining regression modeling in Python can yield valuable insights for pricing strategy and cost optimization.*

---
