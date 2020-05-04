# Supervised Learning in Data Mining

Boston Housing dataset is used to fit **regression models** while Bankruptcy dataset is used for **classification models**.

Regression Models:
  1. Linear Regression
  2. Regression Tree
  3. Bagging
  4. Random Forest
  5. Boosting
  6. GAM
  7. Neural Network

Classification Models:
  1. Logisitic Regression
  2. Classification Tree
  3. GAM
  4. Neural Network

## Boston Housing Data

### Executive Summary

The Boston Housing dataset consisting of 506 observations with 14 variables. The response variable is **medv**, the median housing price which is continuous in nature. Using a 70-30 sample split we fit multiple models on the training data and compare the In-sample and Out-sample Average Sum Square Errors (ASE). The table below summarizes the methods:

|Sr.|Type|ASE|Out-Sample|
|:-:|:--:|:-:|:--------:|
|1|Linear Regression|23.597306|18.688102|
|2|Regression Tree|16.385094|23.013215|
|3|Bagging|12.959958|10.727727|
|4|Random Forest|12.74322|7.157639|
|5|Boosting|2.10690|6.924797|
|6|GAM|8.432493|16.161948|
|7|Neural Network|6.859694|11.741971|

For the Linear Regression model we fit using AIC and BIC criterion and try LASSO variable selection. The best MSE and R<sup>2</sup><sub>adj</sub> is found in the AIC model.  
A Regression tree is fit since the response is continuous. An optimal tree with 6 terminal nodes was built.   
Advanced trees such as Bagging, Random Forest and Boosting are also fit and optimized.  
Boosting shows the best performance in-sample and out-of-sample.  
A General Additive Model is fit with **chas** and **rad** as linear terms and splines for all other terms and **age**, **black** and **zn** were not significant so were removed.  
A Neural network with 2 hidden layers each having 3 nodes was found to show least variation on error between train and test set.  

### Exploratory Data Analysis

Split the data as 70% training and 30% testing.
For all model training we will stick only to training set.

### Linear Regression

Stepwise with AIC and BIC as criterion and LASSO regression models are fit.  
The lowest MSE and best R<sup>2</sup><sub>adj</sub> is seen with stepwise AIC regression.  

```medv = 36.51 - 0.57*lstat + 3.72*rm - 0.87*ptratio - 1.7*dis - 17.69*nox + 0.01*black + 0.34*rad + 2.67*chas + 0.05*zn - 0.01*tax - 0.1*crim```

### Regression Tree

The response is numerical (continuous) so we fit a regression tree.  
Plotting the complexity paramater (cp) we find the lowest cp value within 1 se is 0.03.  
Using cp = 0.03 to prune the tree we get 6 terminal nodes.

### Bagging

Bagging (Bootstrap and Aggregating) helps to improve prediction accuracy.  
It fits a tree for each bootsrap sample, and then aggregates the predicted values from all trees.  
Plotting the errors against number of trees indicates error flattens after fitting 200 trees.  

### Random Forest

Random Forest is similar to bagging except that we randomly select 'm' out of 'p' predictors as candidate variables for each split in each tree.  
In the case of regression trees the default m = p/3.  

We fit trees using Random Forest with default 500 trees.  
We find that **lstat** and **rm** are the most important factors.  
Plotting the Out-of_bag error for these 500 trees shows us that the error seems to flatten after 100 trees.

### Boosting Tree

Boosting sequentially builds a number of small trees, and each time, the response is the residual from last tree.  
We choose to fit 5000 trees with a depth of 8 splits, shrinkage parameter = 0.01 and doing a 3-fold CV.  
We find that **lstat** and **rm** are the most important factors.

Plotting the error against trees for the CV error shows optimal trees required to be 1223  
We fit a second boosting model with 1223 trees to avoid overfitting the data.

### GAM

Generalized Additive Models use a sum of non-paramteric functions over each component of X to capture non-linear relationships.  
**rad** and **chas** are not included in the spline terms as they are quasi-categorical and categorical respectively.  
Each term was checked for significance at 0.05 level vefore removal and if the edf was ~1 it was moved to be a linear term.  
The removal of **age**, **black** and **zn** reduced the GCV score to 11.587 and so that model was selected.

### Neural Network

The dependent variable is numeric so we set linear.output to be true.  
Multiple models were fit:

  1. 1-hidden layer - vary nodes from 1 to 9  
  2. 2-hidden layers - Vary nodes from 1 to 9 in first layer and vary nodes from 1 to 5 in second layer

The model with 3 nodes in the first layer and 3 nodes in second layer had lower out-sample error and lower variance between in-sample and out-sample errors and so it was picked.

## Bankruptcy Data

### Executive Summary

The Bankruptcy dataset contains 5436 observations and 13 variables. R1 to R10 are continuous variables contain financial information which will be used as predictor variables, DLRSN is a binary variable where 0 labels not bankrupt, 1 labels bankrupt. Using a 70-30 sample split we fit multiple models on the training data and compare the asymmetric misclassification cost (AMC) using the cut-off probability as 1/36.  
The table below summarizes the methods:

|Sr.|Type|In-sample AMC|Out-Sample AMC|
|:-:|:--:|:-----------:|:------------:|
|1|Logistic Regression|0.6651|0.6689|
|2|Classification Tree|0.5007|0.6879|
|3|GAM|0.5719|0.6419|
|4|Neural Network|0.4896|0.5984|

For the logistic regression model AIC criterion was slightly better than BIC and so was picked with out-sample TPR of 96.04%.    
A classification tree with 14 terminal nodes was pruned to have 8 terminal nodes with out-sample TPR of 95.15%.  
The GAM model was built with **R1** and **R4** as linear terms and the rest having splines functions which gave out-sample TPR of 95.60%.  
Finally a Neural Network model was fit with one hidden layer having 7 nodes and decay rate of 0.25 to avoid overfitting. The out-sample TPR was found to be 95.60%.   

### Exploratory Data Analysis

Split the data as 70% training and 30% testing.  
For all model training we will stick only to training set.  
In the training set 14.4% of the data has DLRSN=1 which are the cases idicating bankruptcy.  
This implies that the data is imbalanced.  

### Logistic Regression

A full model with variables R1 through R10 was fit as a generalized linear model and another null model was fit with only the intercept.  
These two models were used as the upper and lower scopes for finding a best model using AIC and BIC criterions.  
AUC for AIC model (0.882) was slightly better that BIC model (0.88) both being higher than industrial standard of 0.7.  
Since this is imbalanced data we plot the precision-recall curve and see the AUC for AIC model (0.5723) is better than AUC for BIC model (0.5718).  
The asymmetric misclassifaction cost is 0.6652 with:  
Accuracy = 50.46%  
True Positive Rate = 96.54%  
False Positive Rate = 57.31% 

### Classification Tree

All the variables from R1 to R10 were used to fit a classification tree as the response is categorical.  
The tree is allowed to grow till cp=0.0075 and then pruned at 0.01 based on plotcp curve.  
The asymmetric misclassifaction cost is 0.5007 with:  
Accuracy = 56.19%  
True Positive Rate = 98.72%  
False Positive Rate = 51.00%  

### GAM

All the variables R1 to R10 are first fitted with splines which show edf=1 for R1 and R4 so we move those to linear terms.  
The models deviance explained is low.  
The asymmetric misclassifaction cost is 0.5718 with:  
Accuracy = 55.32%  
True Positive Rate = 97.44%  
False Positive Rate = 51.78% 

### Neural Network

The dependent variable is categorical.  
Using the nnet function we try different sizes and see that with get good in-sample and out-sample asymetric costs with size=7.  
The asymmetric misclassifaction cost is 0.4896 with:  
Accuracy = 61.76%  
True Positive Rate = 97.81%  
False Positive Rate = 44.32% 

