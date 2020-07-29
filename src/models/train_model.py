# -*- coding: utf-8 -*-
'''
Trains the example run of the OOS Markov-Switching Autoregression forecasting on Federal Reserve data.

Note that the t/t+1 prediction timing conventions are *not* the exact same between statsmodels' Autoregression,
Markov Autoregression, and Exponential Smoother, so walk-forward validation is handled in separate methods for the 3.
'''

import io
import logging
import pickle
from pathlib import Path
import boto3
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import pandas as pd
from plot_metric.functions import BinaryClassification
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

BUCKET = 'heart-disease-1301' # s3 bucket name
TEST_PERCENT = 0.2 # train / test split

class HeartDiseaseModels():

    '''Loads in features, tries multiple classifiers using recursive feature selection.
    Cross-validation for hyperparameter optimization. Analyzes feature importance. Saves models.'''

    def __init__(self, logger):

        self.logger = logger
        sns.set(style="white")

        self.s3_client = boto3.client('s3')
        self.graphics_path = Path(__file__).resolve().parents[2].joinpath('reports').resolve().joinpath('figures').resolve()
        self.data_path = Path(__file__).resolve().parents[2].joinpath('data').resolve().joinpath('processed').resolve()
        self.models_path = Path(__file__).resolve().parents[2].joinpath('models').resolve()

    def get_data(self):

        '''Reads in csv from s3'''

        obj = self.s3_client.get_object(Bucket=BUCKET, Key='features.csv')
        self.features_df = pd.read_csv(io.BytesIO(obj['Body'].read()))
        self.logger.info('loaded data...')

    def save_model(self, pkl_name, item):

        '''Helper function for saving the model after train/val'''

        pth = Path(self.models_path, pkl_name).with_suffix('.pkl')
        with open(pth, 'wb') as handle:
            pickle.dump(item, handle)

    def prep_data_pipelines(self):

        '''Splits the data up front for k-fold cross validation, builds pipelines.'''

        X, y = self.features_df[[c for c in self.features_df.columns if c != 'target']], self.features_df['target']
        X = X.iloc[:, 1:]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=TEST_PERCENT, random_state=42, shuffle=True)

        self.logreg_pipe = Pipeline(
            [
                ('scaler', StandardScaler()),
                ('feature_selector', SelectFromModel(LinearSVC(random_state=42), threshold=-np.inf, max_features=20)),
                ('logreg', LogisticRegression(random_state=42))
            ]
        )
        # Parameters of pipelines
        self.logreg_param_grid = {
            'logreg__C': np.logspace(-4, 4, 4),
        }
       

    def run_pipelines(self):

        '''Runs the cross-validation training of the pipelines set up, reports best params.'''
        
        search = GridSearchCV(self.logreg_pipe, self.logreg_param_grid, n_jobs=-1)
        search.fit(self.X_train, self.y_train)
        print("Best parameter (CV score=%0.3f):" % search.best_score_)
        print(search.best_params_)
        self.best_logreg = search.best_estimator_
    
    def classification_reports(self):

        '''Post training, creates error reports and confusion matrices.'''

        print('+++++++++++++++++++++++++++++++++++')
        print('LOGISTIC REGRESSION ERRORS: TEST SET')

        self.logreg_yhat_test = self.best_logreg.predict(self.X_test)
        self.logreg_proba_test = self.best_logreg.predict_proba(self.X_test)

        print(classification_report(self.logreg_yhat_test, self.y_test))
        print('Accuracy score: %0.3f' % accuracy_score(self.logreg_yhat_test, self.y_test))
    
    def plot_confusion_matrix(self, y_true, y_pred, classes, name, normalize=False, title=None, cmap='bwr'):

        """
        Plots the confusion matrix. Normalization can be applied by setting `normalize=True`.
        """

        if not title:
            if normalize:
                title = 'Normalized confusion matrix'
            else:
                title = 'Confusion matrix, without normalization'

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        fig, ax = plt.subplots(dpi=80)
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # Decorations
        ax.set(xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=classes, yticklabels=classes,
            title=title,
            ylabel='True label',
            xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        pth = Path(self.graphics_path, 'conf_matrix_'+name).with_suffix('.png')
        plt.savefig(pth)
    
    def plot_roc_curve(self, name):

        '''Plots the ROC curve for given model (by definition requires a model that can gen probabilities)'''

        # Visualisation with plot_metric
        bc = BinaryClassification(self.y_test, self.logreg_proba_test[:,1], labels=["No Disease", "Heart Disease"])

        # Figures
        plt.figure(figsize=(6, 6))
        bc.plot_roc_curve()
        pth = Path(self.graphics_path, 'roc_curve_'+name).with_suffix('.png')
        plt.savefig(pth)

    def execute_analysis(self):

        '''Runs the necessary methods'''

        self.get_data()
        self.prep_data_pipelines()
        self.run_pipelines()
        self.classification_reports()
        self.plot_confusion_matrix(
            y_true=self.y_test,
            y_pred=self.logreg_yhat_test,
            classes=['No Disease', 'Heart Disease'],
            normalize=True,
            name='logreg',
            title='Confusion Matrix: Logistic Regression'
        )
        self.plot_roc_curve(name='logreg')

def main():

    """ Runs training of models and hyperparameter tuning, saves graphs, logs progress."""

    logger = logging.getLogger(__name__)
    logger.info('running models...')
    Classification_Models = HeartDiseaseModels(logger)
    Classification_Models.execute_analysis()

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
