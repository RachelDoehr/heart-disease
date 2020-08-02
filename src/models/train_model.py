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
from sklearn.ensemble import VotingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from plot_metric.functions import BinaryClassification
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import cm

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
        # Parameters of pipelines
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
        self.logger.info('built training and scaling pipelines...')

    def train_pipelines(self):

        '''Runs the cross-validation training of the pipelines set up, reports best params.'''

        self.best_models = []
        
        for m, p in zip([self.logreg_pipe, self.rf_pipe, self.ada_pipe], [self.logreg_param_grid, self.rf_param_grid, self.ada_param_grid]):

            search = GridSearchCV(m, p, n_jobs=-1)
            search.fit(self.X_train, self.y_train)
            print("Best parameter (CV score=%0.3f):" % search.best_score_)
            print(search.best_params_)

            self.best_models.append(search.best_estimator_)
            self.logger.info('trained one grid search for one pipeline...')

    def compile_voting_classifier(self):
    
        '''Given the trained models, put together a soft voting classifier to balance individual model
        weaknesses but overall relatively comparable performance.'''

        self.eclf = VotingClassifier(
            estimators=[
                ('logreg', self.best_models[0]),
                ('rf', self.best_models[1]),
                ('bdt', self.best_models[2])
            ], voting='soft', weights=[2, 1, 1])

        self.eclf = self.eclf.fit(self.X_train, self.y_train)
        self.best_models.append(self.eclf)

    def classification_reports(self):

        '''Post training, creates error reports and confusion matrices.'''

        self.yhats_test = []
        self.probas_test = []

        for m, l in zip(self.best_models, ['Logistic Regression', 'Random Forest', 'AdaBoost Decision Trees', 'Meta Voting Classifier']):
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
        plt.close()
    
    def plot_roc_curve(self, yproba, name):

        '''Plots the ROC curve for given model (by definition requires a model that can gen probabilities)'''

        # Visualisation with plot_metric
        bc = BinaryClassification(self.y_test, yproba, labels=["No Disease", "Heart Disease"])

        # Figures
        plt.figure(figsize=(6, 6))
        bc.plot_roc_curve()
        pth = Path(self.graphics_path, 'roc_curve_'+name).with_suffix('.png')
        plt.savefig(pth)
        plt.close()

    def gen_error_graphics(self):

        for y, l in zip(self.yhats_test, ['Logistic_Regression', 'Random_Forest', 'AdaBoost_Decision_Trees', 'Meta Voting Classifier']):

            self.plot_confusion_matrix(
                y_true=self.y_test,
                y_pred=y,
                classes=['No Disease', 'Heart Disease'],
                normalize=True,
                name=l,
                title='Confusion Matrix: Test Set'
            )
        for p, l in zip(self.probas_test, ['Logistic_Regression', 'Random_Forest', 'AdaBoost_Decision_Trees', 'Meta Voting Classifier']):
            
            self.plot_roc_curve(name=l, yproba=p[:, 1])

        self.logger.info('plotted confusion matrices and ROC curves in /reports/figures/...')

    def create_avg_people(self):

        '''Prep work to do analysis on the learned decision space for age/cholesterol. Creates
        a 'typical' person in the dataset for each gender. Handles categorical variables as modes.'''

        categoricals = self.X_train[[c for c in self.X_train.columns if '__' in c or c in ['sex', 'exang']]]
        continous = self.X_train[['trestbps', 'chol', 'thalach', 'oldpeak', 'age', 'sex']]

        continous_avg_sex = continous.groupby('sex').mean().reset_index()
        categoricals_mode_sex = categoricals.groupby('sex').sum().reset_index()

        # create the synthetic categorical variables by sex
        mode_out_df = categoricals_mode_sex.copy()
        for col in mode_out_df.columns:
            if col != 'sex':
                mode_out_df[col].values[:] = 0

        def _find_mode(variable, in_df, out_df):

            '''Helper func for finding mode of a variable and updating'''
            mode_col = in_df[[c for c in in_df.columns if variable+'__' in c]].idxmax(axis=1)
            out_df[mode_col.values[0]].values[:] = 1

        def _find_sex_modes(s):
            
            '''Helper func for executing mode analysis'''
            s_df = categoricals_mode_sex[categoricals_mode_sex.sex == s]
            s_out_df = mode_out_df[mode_out_df.sex == s]
            # ---- iterate through the possible categorical variables
            vars = ['cp', 'thal', 'restecg', 'slope', 'ca']
            [_find_mode(v, s_df, s_out_df) for v in vars]
            
            a = continous_avg_sex.merge(s_out_df, how='left', left_on='sex', right_on='sex')
            return a
        
        s_0, s_1 = [_find_sex_modes(ss) for ss in [0, 1]]
        
        self.avg_people = pd.concat([s_0, s_1], ignore_index=True).dropna(how='any')

    def create_synthetic_people(self):

        '''Helper function to generate synthetic data that uses the avg/mode for all other variables
        than resting heart rate and age.'''

        self.create_avg_people()

        def _gen_var_range(v, gender):
    
            '''Helper func to generate synthetic data of possible range of var in data'''

            s_df = self.avg_people[self.avg_people.sex == gender]
            min_trestbps = int(np.round(self.X_train[self.X_train.sex == gender][v].min(), 0))
            max_trestbps = int(np.round(self.X_train[self.X_train.sex == gender][v].max(), 0))

            s_df = s_df.append([s_df]*(max_trestbps - min_trestbps), ignore_index=True)
            s_df[v] = pd.Series(np.arange(min_trestbps, max_trestbps, 1))
            
            return s_df.dropna(how='any')

        s_0_trestbps, s_1_trestbps = [_gen_var_range('trestbps', s) for s in [0, 1]] # heart rate
        s_0_age, s_1_age = [_gen_var_range('chol', s) for s in [0, 1]] # cholesterol
        
        s_0_age['key'] = 0
        s_0_trestbps['key'] = 0
        full_range_0 = s_0_age.merge(s_0_trestbps[['key', 'trestbps']], on='key', how='outer')
        full_range_0['trestbps'] = full_range_0['trestbps_y']

        s_1_age['key'] = 0
        s_1_trestbps['key'] = 0
        full_range_1 = s_1_age.merge(s_1_trestbps[['key', 'trestbps']], on='key', how='outer')
        full_range_1['trestbps'] = full_range_1['trestbps_y']

        self.synthetic_male, self.synthetic_female = full_range_0, full_range_1

    def map_decision_space(self, name, mdl_id):

        '''Visualize the decision boundary for the models with respect to a few key variables of interest,
        blood pressure, age, and sex. Generates synthetic data to completely map the full range of possible
        values blood pressure and age can take on (within the range of the dataset which is a limitation on life).
        
        Synthetic data holds other variables at the mean or mode (if categorical) levels. Then the synthetic data is
        pushed through the trained models to generate predictions. Predicted probabilities of heart disease are plotted.'''

        self.create_synthetic_people()
        sns.set(style='white')
    
        X_0 = self.synthetic_female[self.X_train.columns]
        X_0['p_disease'] = self.best_models[mdl_id].predict_proba(X_0)[:, 1]
        df = X_0[['p_disease', 'chol', 'trestbps']]
        yhat_0 = self.best_models[mdl_id].predict_proba(self.X_train[self.X_train.sex == 1])[:, 1]

        # plot probability surface
        fig = plt.figure(dpi=100)
        ax1 = fig.gca(projection='3d')

        ax1.plot_trisurf(df['chol'], df['trestbps'], df['p_disease'], cmap=plt.cm.coolwarm, linewidth=0.001, alpha=0.85)
        ax1.scatter(self.X_train[self.X_train.sex == 1]['chol'], self.X_train[self.X_train.sex == 1]['trestbps'], yhat_0,
        c='black', s=30)

        # Decorations
        ax1.set_xlabel('Cholesterol', fontsize=9)
        ax1.tick_params(axis='x', rotation=0, labelsize=8)
        ax1.set_ylabel('Resting Heart Rate', color='black', fontsize=9)
        ax1.tick_params(axis='y', rotation=0, labelsize=8)
        ax1.set_zlabel('Probability of Heart Disease', color='black', fontsize=9)
        ax1.tick_params(axis='z', rotation=0, labelsize=8)

        plt.title('Learned Decision Space: '+name, fontsize=12, fontweight='bold')
        pth = Path(self.graphics_path, 'probability_plot_'+name).with_suffix('.png')
        plt.savefig(pth)
        plt.show()
        plt.close()
        
        self.logger.info('plotted and save probability plot for one model in /reports/figures...')

    def execute_analysis(self):

        '''Runs the necessary methods'''

        self.get_data()
        self.prep_data_pipelines()
        self.train_pipelines()
        self.compile_voting_classifier()
        self.classification_reports()
        self.gen_error_graphics()
        self.map_decision_space(name='logreg', mdl_id=0)
        #self.map_decision_space(name='random_forest', mdl_id=1)
        #self.map_decision_space(name='adaboost_trees', mdl_id=2)

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
