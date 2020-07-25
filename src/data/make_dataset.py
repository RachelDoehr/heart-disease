 #-*- coding: utf-8 -*-

'''
Preliminary data cleaning and visualization/exploration.
'''

from pathlib import Path
from io import StringIO
import logging
from dotenv import find_dotenv, load_dotenv
import boto3
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import gridspec
from cycler import cycler
import datetime
import io
from functools import reduce
import pandas as pd
import numpy as np
from pandas.plotting import register_matplotlib_converters

BUCKET = 'heart-disease-1301' # s3 bucket name

class DatasetMaker():

    '''Loads up train and test dataset stored in s3. Data from UCI ML database.
    Performs initial data aggregation and exploratory visualizations.

    Feature creation and saves copy of features locally and to s3.'''
    
    def __init__(self, logger):

        self.logger = logger
        sns.set(style="white")
        register_matplotlib_converters()
        self.s3_client = boto3.client('s3')
        self.graphics_path = Path(__file__).resolve().parents[2].joinpath('reports').resolve().joinpath('figures').resolve()
        self.data_path = Path(__file__).resolve().parents[2].joinpath('data').resolve().joinpath('processed').resolve()

    def get_data(self):

        '''Reads in csv from s3'''
        obj = self.s3_client.get_object(Bucket=BUCKET, Key='uci_heart.csv')
        self.raw_df = pd.read_csv(io.BytesIO(obj['Body'].read()))
        self.logger.info('loaded data, now cleaning...')

    def visualize_continuous(self):

        '''Explore the distributions of the continous variables'''

        continous_features = [
            ['age', 0, 0, 'Age'],
            ['trestbps', 0, 1, 'Resting Blood Pressure'],
            ['chol', 1, 0, 'Cholesterol (mg/dl)'],
            ['thalach', 1, 1, 'Maximum Heart Rate (bpm)'],
            ['oldpeak', 2, 0, 'ST Depression in Exercise Relative to Rest']
        ]
        # Draw
        plt.figure(figsize=(10, 10), dpi=80)
        plt.suptitle('Heart Disease Predictors: Distributions of Continous Variables', y=0.9, fontsize=14)
        gs = gridspec.GridSpec(3, 2)

        for idx, v in enumerate(continous_features):
            
            plt.subplot(gs[v[1], v[2]])
            ax = sns.distplot(self.raw_df[v[0]], color="deepskyblue", label=v[3])
            plt.xlabel(v[3], fontsize=8)
            plt.ylabel('Kernel density estimation', fontsize=8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        pth = Path(self.graphics_path, 'continous_variables_dist').with_suffix('.png')
        plt.savefig(pth)
 
    def visualize_categorical(self):

        '''Explore the distributions of categorical and binary variables'''

        categorical_features = [
            ['sex', 0, 0, 'Sex'],
            ['target', 0, 2, 'Heart Disease'],
            ['cp', 0, 1, 'Chest Pain Type'],
            ['ca', 1, 0, 'No of Major Vessels Colored by Flouroscopy'],
            ['thal', 1, 1, 'Thalassemia'],
            ['fbs', 2, 0, 'Fasting Blood Sugar (>120mg/dl)'],
            ['restecg', 2, 1, 'Resting ECG Results'], 
            ['exang', 2, 2, 'Exercise Induced Angia'],
            ['slope', 1, 2, 'Slope of Peak Exercise ST Segment']
        ]
        # Draw
        plt.figure(figsize=(12, 12), dpi=80)
        plt.suptitle('Heart Disease Predictors: Distributions of Categorical Variables', y=0.9, fontsize=14)
        gs = gridspec.GridSpec(3, 3)
        
        for idx, v in enumerate(categorical_features):
            
            plt.subplot(gs[v[1], v[2]])
            if v[0] == 'target':
                ax = sns.countplot(self.raw_df[v[0]], color='gray')
            else:
                ax = sns.countplot(self.raw_df[v[0]], color='deepskyblue', alpha=0.6)
            plt.xlabel(v[3], fontsize=8)
            plt.ylabel('Total Count', fontsize=8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        pth = Path(self.graphics_path, 'categorical_variables_dist').with_suffix('.png')
        plt.savefig(pth)

    def visualize_scatter(self):

        '''Pairwise correlations visually by target type'''

        c_vars = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        colMap={0: "gray", 1: "blue"}
        colors=list(map(lambda x:colMap.get(x), self.raw_df.target))

        # Create a scatter matrix from the dataframe, color by y_train
        grr = pd.plotting.scatter_matrix(self.raw_df[c_vars], c=colors, figsize=(10, 10), marker='o',
                                        hist_kwds={'bins': 20}, s=5)
        plt.legend()
        plt.suptitle('Pairwise Scatterplot of Continous Variables by Target')
        pth = Path(self.graphics_path, 'categorical_variables_scatterpair').with_suffix('.png')
        plt.savefig(pth)

    def explore_correlations(self):

        '''Visualize correlations of post-transformed data'''
        correl = self.transformed_df.iloc[1:,60:]
        corr = correl.corr()
        # generate a mask for the upper triangle
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True

        f, ax = plt.subplots(figsize=(17, 17))

        # generate a custom diverging colormap
        cmap = sns.diverging_palette(10, 220, as_cmap=True)

        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(corr, mask=mask, cmap=cmap, center=0, ax=ax,
                    square=True, linewidths=.2, cbar_kws={"shrink": 0.5})
        pth = Path(self.graphics_path, 'correlation_matrix_1').with_suffix('.png')
        f.savefig(pth)
        self.logger.info('plotted and save figures in /reports/figures/')

    def final_prep_and_save(self):
        '''Removes the columns with significant null values. Can also remove outlier series, nonborrowed reserves.
        Then saves copies.'''
        self.logger.info('removing nulls and outliers, saving...')
        to_remove = ['ACOGNO', 'S&P PE ratio', 'TWEXAFEGSMTHx', 'UMCSENTx'] # leaving reserves in for now

        self.features_df = self.transformed_df.drop(to_remove, axis=1).iloc[1:, :]

        pth = Path(self.data_path, 'features').with_suffix('.csv')
        self.features_df.to_csv(pth)
        # upload to s3
        csv_buffer = StringIO()
        self.features_df.to_csv(csv_buffer)
        s3_resource = boto3.resource('s3')
        s3_resource.Object(BUCKET, 'features.csv').put(Body=csv_buffer.getvalue())

    def execute_dataprep(self):

        self.get_data()
        #self.visualize_continuous()
        #self.visualize_categorical()
        self.visualize_scatter()
        #self.explore_correlations()
        #self.final_prep_and_save()

def main():
    """ Runs data processing scripts to turn raw data from s3 into
        cleaned data ready to be used in model (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    HeartData = DatasetMaker(logger)
    HeartData.execute_dataprep()

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
