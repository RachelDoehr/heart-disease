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
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
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
                ('logreg', LogisticRegression(random_state=42, max_iter=200))
            ]
        )
        self.rf_pipe = Pipeline(
            [
                ('scaler', StandardScaler()),
                ('rf', RandomForestClassifier(random_state=42))
            ]
        )
        self.ada_pipe = Pipeline(
            [
                ('scaler', StandardScaler()),
                ('ada', AdaBoostClassifier())
            ]
        )
        self.ada_pipe_logreg = Pipeline(
            [
                ('scaler', StandardScaler()),
                ('ada', AdaBoostClassifier())
            ]
        )
        # Parameters of pipelines

        # auto generate a range of base models for feature optimization
        base_logreg_estimators = []

        for jj in range(2, 21):
            base_logreg_estimators.append(LogisticRegression(max_iter=jj, random_state=42))
        
        self.logreg_param_grid = {
            'logreg__C': np.logspace(-4, 4, 4),
        }
        self.rf_param_grid = {
            'rf__n_estimators': [int(x) for x in np.linspace(start=200, stop=1000, num=10)],
            'rf__max_depth': [int(x) for x in np.linspace(10, 110, num=11)],
            'rf__bootstrap': [True, False]
        }
        # auto generate a range of base decision stumps
        base_estimators = []
        for jj in range(1, 111, 11):
            base_estimators.append(DecisionTreeClassifier(max_depth=jj, min_samples_leaf=1, random_state=42))
        
        self.ada_param_grid = {
            'ada__base_estimator': base_estimators,
            'ada__learning_rate': np.logspace(-4, 4, 4), # effectively regularization
            'ada__n_estimators': [int(x) for x in np.linspace(start=200, stop=1000, num=10)],
            'ada__algorithm': ['SAMME', 'SAMME.R']
        }
        self.ada_logreg_param_grid = {
            'ada__base_estimator': base_logreg_estimators,
            'ada__learning_rate': np.logspace(-4, 4, 4), # effectively regularization
            'ada__n_estimators': [int(x) for x in np.linspace(start=200, stop=1000, num=10)],
            'ada__algorithm': ['SAMME', 'SAMME.R']
        }
        self.logger.info('built training and scaling pipelines...')

    def train_pipelines(self):

        '''Runs the cross-validation training of the pipelines set up, reports best params.'''

        self.best_models = []
        
        for m, p in zip(
                [
                    self.logreg_pipe,
                    self.rf_pipe, 
                    self.ada_pipe_logreg,
                    self.ada_pipe
                ],
                [
                    self.logreg_param_grid,
                    self.rf_param_grid,
                    self.ada_logreg_param_grid,
                    self.ada_pipe
                ]
            ):

            search = GridSearchCV(m, p, n_jobs=-1)
            search.fit(self.X_train, self.y_train)
            print("Best parameter (CV score=%0.3f):" % search.best_score_)
            print(search.best_params_)

            self.best_models.append(search.best_estimator_)
            self.logger.info('trained one grid search for one pipeline...')
    
    def classification_reports(self):

        '''Post training, creates error reports and confusion matrices.'''

        self.yhats_test = []
        self.probas_test = []

        for m, l in zip(self.best_models, ['Logistic Regression', 'Random Forest', 'AdaBoost LogRegs', 'AdaBoost Decision Stumps']):
            print('+++++++++++++++++++++++++++++++++++')
            print('ERRORS: TEST SET')
            print('Model: ', l)

            self.yhats_test.append(m.predict(self.X_test))
            self.probas_test.append(m.predict_proba(self.X_test))

            print(classification_report(m.predict(self.X_test), self.y_test))
            print('Accuracy score: %0.3f' % accuracy_score(m.predict(self.X_test), self.y_test))
    
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
    
    def plot_roc_curve(self, yproba, name):

        '''Plots the ROC curve for given model (by definition requires a model that can gen probabilities)'''

        # Visualisation with plot_metric
        bc = BinaryClassification(self.y_test, yproba, labels=["No Disease", "Heart Disease"])

        # Figures
        plt.figure(figsize=(6, 6))
        bc.plot_roc_curve()
        pth = Path(self.graphics_path, 'roc_curve_'+name).with_suffix('.png')
        plt.savefig(pth)

    def gen_error_graphics(self):

        for y, l in zip(self.yhats_test, ['Logistic_Regression', 'Random_Forest', 'AdaBoost_LogRegs', 'AdaBoost_Decision_Stumps']):

            self.plot_confusion_matrix(
                y_true=self.y_test,
                y_pred=y,
                classes=['No Disease', 'Heart Disease'],
                normalize=True,
                name=l,
                title='Confusion Matrix: Test Set'
            )
        for p, l in zip(self.probas_test, ['Logistic_Regression', 'Random_Forest', 'AdaBoost_LogRegs', 'AdaBoost_Decision_Stumps']):
            
            self.plot_roc_curve(name=l, yproba=p[:, 1])

        self.logger.info('plotted confusion matrices and ROC curves in /reports/figures/...')
    
    def analyze_feature_importance(self, model):

        '''Given the optimal trained model, use permutations to examine relative feature contribution.
        Check for high multicollinearity which would make correlated features look less importance even if they are.'''

        correl = self.X_test
        corr = correl.corr()
        # generate a mask for the upper triangle
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True

        f, ax = plt.subplots(figsize=(12, 12))

        # generate a custom diverging colormap
        cmap = sns.cubehelix_palette(8, start=.5, rot=-.75)

        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(corr, mask=mask, cmap=cmap, center=0, ax=ax,
                    square=True, linewidths=.2, cbar_kws={"shrink": 0.5})
        pth = Path(self.graphics_path, 'correlation_matrix').with_suffix('.png')
        f.savefig(pth)
        # ----------------------- feature importance on test set
        result = permutation_importance(model, self.X_test, self.y_test, n_repeats=10,
                                random_state=42, n_jobs=2)
        sorted_idx = result.importances_mean.argsort()

        fig, ax = plt.subplots(figsize=(6, 6))
        bp = ax.boxplot(result.importances[sorted_idx].T,
                vert=False, labels=self.X_test.columns[sorted_idx], patch_artist=True)
        
        ## decorations
        for box in bp['boxes']:
            box.set(color='#abfc90', linewidth=2)
            box.set(facecolor='#43ff56', alpha=0.5)

        for whisker in bp['whiskers']:
            whisker.set(color='#abfc90', linewidth=2)

        for cap in bp['caps']:
            cap.set(color='#abfc90', linewidth=2)

        for median in bp['medians']:
            median.set(color='#F90031', linewidth=2)

        for flier in bp['fliers']:
            flier.set(marker='o', color='#f92e4a', alpha=0.5)
        ax.set_title("Permutation Importances (test set)")
        fig.tight_layout()
        plt.show()

        self.logger.info('plotted and save feature correlations and importance in /reports/figures/')

    def execute_analysis(self):

        '''Runs the necessary methods'''

        self.get_data()
        #self.prep_data_pipelines()
        #self.train_pipelines()
        #self.classification_reports()
        #self.gen_error_graphics()
        self.analyze_feature_importance(model='i')

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
