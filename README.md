# Regression Project and R-Programming Report
![Last Commit](https://img.shields.io/github/last-commit/JPP-J/regression_project?style=flat-square)
![Python](https://img.shields.io/badge/Python-100%25-blue?style=flat-square)
![Languages](https://img.shields.io/github/languages/count/JPP-J/regression_project?style=flat-square)

This repository contains the code and report accompanying Jidapa's *Regression Project and R-Programming Report*, featuring:

- Example Python code for regression in [`main.py`](main.py) or on [Kaggle](https://kaggle.com/code/jidapapooljan/linear-regression), demonstrating a  
  **Report Analyzing Amazon Sales Data Using Linear Regression Techniques to Predict Revenue** [`R_hypothesis_testing_report.pdf`](R_hypothesis_testing_report.pdf).  
  The project covers:
  - Data Pre-processing
  - Feature Selection
  - Data Exploration
  - Model Training
  - Model Evaluation

- An **R programming report** (`R_hypothesis_testing_report.pdf`) discussing hypothesis testing techniques including:
  - Chi-Square Test
  - Logit Model
  - Regression Analysis

## Abstract

This study presents a multiple linear regression model developed to predict total revenue from Amazon sales data, focusing on **unit price** and **unit cost** as primary features. The dataset underwent thorough preprocessing, including feature selection via correlation analysis to ensure significant predictors were included.

A pipeline incorporating scaling and regression was constructed. The model was trained and tested using a 90/10 train-test split. The performance was evaluated with an R-squared value of **0.7451**, indicating a strong relationship between the selected features and total revenue.

These results demonstrate the model's potential for forecasting revenue and provide actionable insights for optimizing pricing and cost strategies in e-commerce. Future work is recommended to improve predictive accuracy through additional feature engineering and more advanced modeling techniques.



## Objective

- Select relevant features/variables to use in the linear regression model.
- Apply the linear regression model to predict total revenue based on unit price and unit cost.



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


