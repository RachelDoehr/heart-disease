
![Alt Text](https://github.com/RachelDoehr/heart-disease/blob/master/reports/figures/example_logistic_reg.gif?raw=true)

# Prediction of Cardiac Diagnosis (Heart Disease) Using Patient Data

 *A demonstration of various classification algorithms on UCI ML dataset*

**COMPARISON OF LOGISTIC REGRESSION, RANDOM FOREST, BOOSTED DECISION TREES, AND A VOTING CLASSIFIER). HYPERPARAMETER TUNING + FEATURE IMPORTANCE ANALYSIS**

> -> Uses the popular UCI heart disease patient dataset available <a href="https://archive.ics.uci.edu/ml/datasets/heart+Disease" target="_blank">here</a>

> -> Models include logistic regression, random forest, boosted decision trees, and a meta voting classifier of all 3; logistic regression has the highest accuracy by a small margin

> -> Post-hoc analysis is done on visualizing the various algorithm's learned decision spaces with respect to how cholesterol and resting heart rate affect the predicted probability of heart disease

![Alt Text](https://github.com/RachelDoehr/forecasting/blob/master/reports/figures/markov_ar_specification.PNG?raw=true)

**Motivation & Project Summary**

This project is a demonstration of using various binary classification algorithms on the UCI heart disease dataset to predict whether or not a patient is likely to have heart disease. The broad approach taken was to use k-fold cross validation to tune the hyperparameters and select the optimal model for each algorithm examine. Then, the out of sample performance was tested on a separate hold out batch, and accuracy reports for misclassifications generated. Finally, the relative feature importance of the optimal models was analyzed using a permutation-based importance estimator. This allows insight into the learned model's key drivers of cardiac disease risk.

The models considered are either semiparametric (Logistic Regression) or entirely non-parametric in nature (Random Forest and AdaBoost + Decision Trees). The sample is conveniently evenly balanced across both classes from the start. , <a href="https://github.com/statsmodels/statsmodels/blob/ebe5e76c6c8055dddb247f7eff174c959acc61d2/statsmodels/tsa/regime_switching/markov_switching.py#L702-L703" target="_blank">it is not yet implemented for MS-AR</a>


The extension is demonstrated on Federal Reserve monthly economic data, showing forecast performance for U.S. unemployment claims in "normal" times (~2000 - 2007) and a crisis/recession (~2008 - 2010). **Consistent with the literature, the MS-AR offers comparable out of sample forecasting for normal times and outperforms in the non-dominant regime.**

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

**Preliminary Data Transforms**

*The OOS MS-AR t+1 forecasts for U.S. unemployment claims perform comparably with an AR and exponential smoothing in normal periods, and outperforms in a recession.*

We begin by transforming the data as suggested by the authors of the dataset to make it stationary https://research.stlouisfed.org/econ/mccracken/fred-databases/. The full dataset contains ~140 different series with corresponding strategies for transformation (differencing, logs, logs+differences, etc.). The data can be rougly categorized into different economic factors. For example, the housing variables contain many housing series:

![Alt Text](https://github.com/RachelDoehr/forecasting/blob/master/reports/figures/time_series_transformed_housing.png?raw=true)

The below graph isolates the chosen variable of interest, unemployment claims. The series is post-making stationary. In the walk-forward validation, the initial training sample is 1970 through late 1990's, while 1998-2007 is considered "normal" economic times, and late 2007-2010 is considered "shock/recession."

![Alt Text](https://github.com/RachelDoehr/forecasting/blob/master/reports/figures/y_var_time_series.png?raw=true)

A visual examination of the lag structure using an ACF plot:
![Alt Text](https://github.com/RachelDoehr/forecasting/blob/master/reports/figures/acf_plot.png?raw=true)
 

### **Cross-Model Comparison in Various Regimes / Time Periods**

- Using the optimal hyperparameters (lags) selected above, the below graph shows the MSE of the OOS forecasts
- The MS-AR extension performs comparably with the other univariate statsmodels OOS methods
- It outperforms in the non-dominant regime, consistent with literature

![Alt Text](https://github.com/RachelDoehr/forecasting/blob/master/reports/figures/error_summary.png?raw=true)

The t+1 forecasts plotted with the actuals show the improved ability to model the more extreme values in '08:
![Alt Text](https://github.com/RachelDoehr/forecasting/blob/master/reports/figures/yhat_y_Classical.png?raw=true)


**Detail on Lag Selection/Parameter Tuning**

The baseline models used are an autoregression and exponential smoothing. The mean squared error for the t+1 forecasts using walk-forward validation across various lag orders are:
![Alt Text](https://github.com/RachelDoehr/forecasting/blob/master/reports/figures/AR_errors.png?raw=true)

Given the above, a 3rd order lag for the AR is used. Holt's Exponential Smoothing baseline model, on the other hand, uses a weighted lag combination. No hyperparameters are needed to be optimized in the statsmodels implementation.

The MS-AR's mse for the same period (using the custom extension to allow for walk-forward validation) is:
![Alt Text](https://github.com/RachelDoehr/forecasting/blob/master/reports/figures/MKV_errors.png?raw=true)

A 3rd order Markov model is used as well.

---


<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p> 