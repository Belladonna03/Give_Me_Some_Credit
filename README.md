# Kaggle DataSet - Give Me Some Credit 📊💳

## Project Description 🚀

This project is dedicated to analyzing and predicting the probability of a customer defaulting on a loan based on historical credit data. The dataset is taken from the "Give Me Some Credit" competition on Kaggle. The goal of the project is to build machine learning models that can predict whether a customer will experience difficulties in repaying a loan within the next two years.

## Data Description 📊

The dataset contains the following columns:

- **SeriousDlqin2yrs**: Target variable. 1 — the customer experienced difficulties in repaying the loan, 0 — did not experience difficulties.
- **RevolvingUtilizationOfUnsecuredLines**: Total balance on credit cards and personal lines of credit except real estate and no installment debt like car loans divided by the sum of credit limits.
- **age**: Age of the borrower in years.
- **NumberOfTime30-59DaysPastDueNotWorse**: Number of times the borrower has been 30-59 days past due but no worse in the last 2 years.
- **DebtRatio**: Monthly debt payments, alimony, living costs divided by monthly gross income.
- **MonthlyIncome**: Monthly income.
- **NumberOfOpenCreditLinesAndLoans**: Number of open loans (installment like car loan or mortgage) and lines of credit (e.g., credit cards).
- **NumberOfTimes90DaysLate**: Number of times the borrower has been 90 days or more past due.
- **NumberRealEstateLoansOrLines**: Number of mortgage and real estate loans including home equity lines of credit.
- **NumberOfTime60-89DaysPastDueNotWorse**: Number of times the borrower has been 60-89 days past due but no worse in the last 2 years.
- **NumberOfDependents**: Number of dependents in the family excluding themselves (spouse, children, etc.).

## Exploratory Data Analysis 🔍

### Summary Descriptive Statistics

| Variable Name | Description | Data Type |
|---------------|-------------|-----------|
| SeriousDlqin2yrs | Person experienced 90 days past due delinquency or worse. | Y/N |
| RevolvingUtilizationOfUnsecuredLines | Total balance on credit cards and personal lines of credit except real estate and no installment debt like car loans divided by the sum of credit limits. | percentage |
| Age | Age of borrower in years. | integer |
| NumberOfTime30-59DaysPastDueNotWorse | Number of times borrower has been 30-59 days past due but no worse in the last 2 years. | integer |
| DebtRatio | Monthly debt payments, alimony, living costs divided by monthly gross income. | percentage |
| MonthlyIncome | Monthly income. | real |
| NumberOfOpenCreditLinesAndLoans | Number of Open loans (installment like car loan or mortgage) and Lines of credit (e.g., credit cards). | integer |
| NumberOfTimes90DaysLate | Number of times borrower has been 90 days or more past due. | integer |
| NumberRealEstateLoansOrLines | Number of mortgage and real estate loans including home equity lines of credit. | integer |
| NumberOfTime60-89DaysPastDueNotWorse | Number of times borrower has been 60-89 days past due but no worse in the last 2 years. | integer |
| NumberOfDependents | Number of dependents in family excluding themselves (spouse, children, etc.). | integer |

### Correlation Heatmap

To examine feature reduction, the following shows a correlation heatmap of the original data. Red implies values are correlated.

![Correlation Heatmap](images/correlation_heatmap.png)

### Class Imbalance

The variable SeriousDlqin2yrs is labeled either 0's and 1's with no other values. The class imbalance is 6.7% (ratio of 1:14) as shown by the following pie chart. As this majority class of 0's could dominate the prediction, increasing the relevant proportion of the minority class of 1's was also tried using downsampling with upsampling of the dominant class.

![Class Imbalance](images/class_imbalance.png)

### Age Variable

For the age variable, it can be believed that the minimum age should start at 21 as there is only one value below this with an age of 0. This value can be adjusted to take the average value of the distribution.

### Outliers and Imputation

For the variables "NumberOfTime30-59DaysPastDueNotWorse", two years divided into 60 day increments suggests a maximum value of about 12-24 and this is what the data set exhibits except for a cluster of points at the values of 96 and 98. These are either strange mistakes or possibly values specific to the dataset with some other meaning such as 'unknown'. Further, the variables "NumberOfTimes90DaysLate" and "NumberOfTime60-89DaysPastDueNotWorse" also contained these 96 and 98 pillars of data which showed up on correlation between the three variables. To accommodate for this, the values were considered to be Winsorized to the previous largest value but the median value was chosen with the aim to have less impact over the sample distribution. After altering the values for 96 and 98, the significant inter-correlation variables NumberOfTime30-59DaysPastDueNotWorse, NumberOfTimes90DaysLate, and NumberOfTime60-89DaysPastDueNotWorse drops away.

### Missing Values

The feature MonthlyIncome has 29,731 missing values or 19.8% of 150,000 examples with the belly of the distribution around 5,000 and a long right-tail.

NumberOfDependents had 3924 missing values or 2.6% of the dataset. These values were set to zero. There is a jump from 6 to 20 dependents with 244 examples in that 99.9% percentile bin. All values were nominal whole numbers such that there were no fractions of a dependent.

## Model Estimations 🤖

### Cross-Validation

Split the training data into 2-fold Cross Validation with 75% random samples in training and the remaining for validation/test. Logistic regression can be considered the base model to compare others to. Out-of-sample fit on unseen data gave an AUC of 0.8117. With the following ROC curve:

![Logistic Regression ROC Curve](images/logistic_regression_roc.png)

### Feature Importance

The important features according to the Logistic model are shown in the following figure:

![Logistic Regression Feature Importance](images/logistic_regression_feature_importance.png)

### XGBoost Model

The XGBoost estimation provided an improvement over Logistic regression, AdaBoost, and standard tree. Generating an AUC of 0.86146 and resulting in the following AUC curve:

![XGBoost ROC Curve](images/xgboost_roc.png)

### Summary Table of Models

| Model | AUC - Training | AUC - Validation |
|-------|----------------|------------------|
| Logistic Regression | 0.816 | 0.813 |
| Decision Tree | 0.618 | 0.601 |
| Random Forest | 0.889 | 0.864 |
| Gradient Boosting | 0.878 | 0.864 |
| XGBoost | 0.881 | 0.866 |
| Stacking | 0.866 | |

## Conclusions 📝

XGBoost seems to perform quite well out of the box as well as Random Forest. It could make sense to further investigate combining these model powers in a well-tuned ensemble or stacking method and even further the impact of a deep learning algorithm. 

A model used in this analysis can be used in an online credit-line approval process as well as provided to product and client relationship managers to detect ongoing potential credit threats. Complex implementation such as those involving stacking with Deep Learning models require time for hyperparameter tuning. Further, the production implementation of these models does somewhat step away from an intuitive approach of regression models such as Logistic regression.

## Future Analysis 🚀

The extreme examples may be the ones which default more and this extreme distribution may be worth estimating separately to the rest of the distributions. Traditionally, Ensemble techniques win Kaggle competitions. This includes ideas such as Stacking. These techniques require Hyper-parameter tuning which can occupy considerable time and was out-of-scope for this exercise for me. But could be something to consider in the future.

## Requirements 🛠️

Python (>3.4), Jupyter, and SciKitLearn.

---

Thank you for your interest in the project! If you have any questions or suggestions, feel free to reach out. 😊
