
![Alt Text](https://github.com/RachelDoehr/heart-disease/blob/master/reports/figures/example_logistic_reg.gif?raw=true)

# Prediction of Cardiac Diagnosis (Heart Disease) Using Patient Data

 *A demonstration of various classification algorithms on UCI ML dataset*

**COMPARISON OF LOGISTIC REGRESSION, RANDOM FOREST, BOOSTED DECISION TREES, AND A VOTING CLASSIFIER). HYPERPARAMETER TUNING + FEATURE IMPORTANCE ANALYSIS**

> -> Uses the popular UCI heart disease patient dataset available <a href="https://archive.ics.uci.edu/ml/datasets/heart+Disease" target="_blank">here</a>

> -> Models include logistic regression, random forest, boosted decision trees, and a meta voting classifier of all 3; logistic regression has the highest accuracy by a small margin

> -> Post-hoc analysis is done on visualizing the various algorithm's learned decision spaces with respect to how cholesterol and resting heart rate affect the predicted probability of heart disease

![Alt Text](https://github.com/RachelDoehr/heart-disease/blob/master/reports/figures/logistic_equation.PNG?raw=true)

**Motivation & Project Summary**

This project is a demonstration of using various binary classification algorithms on the UCI heart disease dataset to predict whether or not a patient is likely to have heart disease. The broad approach taken was to use k-fold cross validation to tune the hyperparameters and select the optimal model for each algorithm examine. Then, the out of sample performance was tested on a separate hold out batch, and accuracy reports for misclassifications generated. Finally, two well-known variables, cholesterol and resting heart rate, are examined by visualizing each model's learned decision space with respect to possible values of those predictors.

The models considered are either semiparametric (Logistic Regression) or entirely non-parametric in nature (Random Forest, AdaBoost + Decision Trees, Voting Classifier). The sample is conveniently evenly balanced across both classes from the start.**The outperformance of the logistic regression relative to random forests and boosted trees implies that 1) the problem is linearly separable in nature, and 2) there are more useful variables in the dataset than noise variables (natural, given the curated nature of this data).**

> ***ReadMe Table of Contents***

- INSTALLATION & SETUP

*RESULTS*
- DATA VISUALIZATION
- ERROR METRICS
- FEATURE IMPORTANCE

---

## Installation & Setup

### Clone

- Clone this repo to your local machine using `https://github.com/RachelDoehr/heart-disease.git`

### Setup

- Install the required packages

> Requires python3. Suggest the use of an Anaconda environment for easy package management.

```shell
$ pip install -r requirements.txt
```

### Example Run of All Processes

- Recommend running train_model as a background process that can be returned to later if running locally. Given the range of hyperparameters tested in cross-validation, significant training time is incurred
- Estimated runtime will vary by computer, but on an Intel(R) Core(TM) i5-6200U CPU @2.30GHz with 8.00 GB memory, searching all parameters for the models takes ~2 hours

```shell
$ python /src/data/build_features.py > /logs/features_log.txt
$ nohup python /src/models/train_model.py > /logs/models_log.txt &
```

---

## Results

**Preliminary Data Visualization**

*The dataset consists of 5 continous variables and the remaining 8 are categorical or binary, which are handled appropriately with dummy variables. The target, 0 or 1, represents whether or not a patient developed heart disease.*

We begin by plotting the raw distributions of the continous variables with histograms:

![Alt Text](https://github.com/RachelDoehr/heart-disease/blob/master/reports/figures/continous_variables_dist.png?raw=true)

Four of the five appear relatively normal distributions, albeit with slight skews. ST depression, however, is not. The same variables' distributions bifurcated by whether or not the patient had heart disease (using a kernel density estimator) are:

![Alt Text](https://github.com/RachelDoehr/heart-disease/blob/master/reports/figures/continous_variables_by_target.png?raw=true)

Some of those variables do appear to significantly vary by target. Additionally, the remaining categorical variables distributions include:
![Alt Text](https://github.com/RachelDoehr/heart-disease/blob/master/reports/figures/categorical_variables_dist.png?raw=true)
 
Additional visualizations are available in /reports/figures.

**Error Metrics**

Grid search k-fold cross-validation is used across a variety of hyperparameters to select the optimal fit for each of the models examined.

| Model                               	| OOS Accuracy 	|
|-------------------------------------	|--------------	|
| Logistic Regression                 	| 88.5%        	|
| Random Forest                       	| 86.9%        	|
| AdaBoost Decision Trees             	| 83.6%        	|
| Voting Classifier of Models (1)-(3) 	| 88.5%        	|

In addition to the simple accuracy above, the confusion matrices for Logistic Regression and Random Forests' performance on the test set are (others available in /reports/figures):

*Logistic Regression*

![Alt Text](https://github.com/RachelDoehr/heart-disease/blob/master/reports/figures/conf_matrix_Logistic_Regression.png?raw=true)

*Random Forest*

![Alt Text](https://github.com/RachelDoehr/heart-disease/blob/master/reports/figures/conf_matrix_Random_Forest.png?raw=true)

Given the above, a 3rd order lag for the AR is used. Holt's Exponential Smoothing baseline model, on the other hand, uses a weighted lag combination. No hyperparameters are needed to be optimized in the statsmodels implementation.

The MS-AR's mse for the same period (using the custom extension to allow for walk-forward validation) is:
![Alt Text](https://github.com/RachelDoehr/heart-disease/blob/master/reports/figures/roc_curve_Logistic_Regression.png?raw=true)

A 3rd order Markov model is used as well.

---


<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p> 