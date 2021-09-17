# Problem 1
Walmart Technology has been tasked with identifying two groups of people for
marketing purposes: People who earn an income of less than $50,000 and those who
earn more than $50,000. To assist in this pursuit, Walmart has developed a means of
accessing 40 different demographic and employment related variables for any person
they are interested in marketing to. Additionally, Walmart has been able to compile a
dataset that provides gold labels for a variety of observations of these 40 variables
within the population. Using the dataset given, train and validate a classifier that
predicts this outcome.

## Project Intro/Objective
The purpose of this project is training and evaluating income classification model as well as using it to
run inference on new observations.

### Pre-processing Approach Used
* delete columns that have too many '?'
* replace '?' with NA for further operation
* replace NA with the most common values of each columns
* convert object value to int
* upsampling

### Model Architecture
* Continuously add trees and continuously perform feature splitting to grow a tree. Each time the model adds a tree, it actually learns a new function f(x) to fit the residual of the last prediction.
When we get k trees after training, we need to predict the score of a sample. In fact, according to the characteristics of this sample, select a corresponding leaf node, and each leaf node corresponds to a score.
In the end, we only need to add up the scores corresponding to each tree to get the predicted value of the sample.

### Training Algorithm
* Xgboost

### Evaluation Procedure
* use the classification report visualizer displays the precision, recall, F1, and support scores for the model
* adjust the upsampling parameters to achieve the best prediction performance

For detailed code and ideas, please check the file 'classification model.html'

### References
* https://scikit-learn.org/stable/modules/classes.html
* https://xgboost.readthedocs.io/en/latest/python/python_api.html
* https://pandas.pydata.org/docs/reference/index.html
* https://www.kdnuggets.com/2017/06/7-techniques-handle-imbalanced-data.html
* https://www.rdocumentation.org/packages/xgboost/versions/1.4.1.1/topics/xgb.train


# Problem 2
Walmart is also interested in developing a rudimentary segmentation model of the
people represented in this dataset in the context of marketing. Using one or more of
your favorite machine learning or data science techniques, create such a segmentation
model and demonstrate how the resulting groups differ from one another.

## Project Intro/Objective
The purpose of this project is generating segmentation model and using it to predict which segment a
new observation will belong to

### Pre-processing Approach Used
* delete columns that have too many '?'
* delete features that would weaken the interpretability of segmentation
* replace '?' with NA for further operation
* replace NA with the most common values of each columns
* convert object value to int
* standardize data
* select the most important features
* upsampling

### Model Architecture
* Logistic regression uses the sigmoid function to transform the output of linear regression to return probability values, and then the probability values can be mapped to two or more discrete classes.
* Random forest is a special bagging method that uses decision trees as a model in bagging. First, use the bootstrap method to generate m training sets. Then, for each training set, construct a decision tree. When the node finds the feature to split, not all the features can be found to maximize the index, but the feature Randomly extract a part of the features and find the optimal solution among the extracted features, and apply it to the node to split. The random forest method has bagging, that is, the idea of integration, which is actually equivalent to sampling both samples and features, so overfitting can be avoided.

### Training Algorithm
* Logistic Regression
* Random Forest

### Evaluation Procedure
* use the classification report visualizer displays the precision, recall, F1, and support scores for the model
* adjust the upsampling parameters and n_estimators of RandomForestClassifier to achieve the best prediction performance

For detailed code and ideas, please check the file 'segmentation model.html'

### References
* https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
* https://www.kaggle.com/faressayah/logistic-regression-data-preprocessing
* https://www.kaggle.com/gilangpanduparase/air-line-customer-segmentation
* https://xgboost.readthedocs.io/en/latest/tutorials/input_format.html
* https://scikit-learn.org/stable/modules/generated/sklearn.compose.make_column_transformer.html
* https://seaborn.pydata.org/api.html
