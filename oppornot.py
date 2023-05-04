import os
import pandas as pd
import numpy as np
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
# from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion
from sklearn.pipeline import Pipeline, make_pipeline
# from itertools import chain
import datetime
import re
from sklearn import svm
from sklearn.model_selection import train_test_split
# from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV
# from sklearn.linear_model import BayesianRidge
from sklearn import metrics
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc, make_scorer, recall_score, accuracy_score, \
    precision_score, confusion_matrix
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import neural_network
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_recall_fscore_support
import statistics
from sklearn import metrics

ROOT_DIR = os.path.abspath(os.curdir)



def oppornot(file):
    def data_preprocessing(file):
        # for files in glob.glob(fil1):
        #     with open(files) as fil1:
        # df = pd.read_csv(fil1)
        # for files in glob.glob(file2):
        #     with open(files) as file2:
        df_trial = pd.read_csv(file)

        # df_original = df
        df_trial_original = df_trial

        # df.drop('Website', axis=1, inplace=True)
        # df.drop('Comment', axis=1, inplace=True)
        # df.drop('Salesforce Status', axis=1, inplace=True)
        # df.tail()
        df_trial.drop('Website', axis=1, inplace=True)
        df_trial.drop('Comment', axis=1, inplace=True)
        df_trial.drop('Salesforce Status', axis=1, inplace=True)

        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)

        scale_mapper = {
            "Negative": 0,
            "Neutral": 0.33,
            "Moderate": 0.67,
            "High": 1
        }

        scale_mapper_company_size = {
            "0-50": 0,
            "51-200": 0.17,
            "201-500": 0.33,
            "501-1000": 0.5,
            "1001-5000": 0.67,
            "5001-10000": 0.84,
            "10001+": 1
        }

        yesno_mapper = {
            "Yes": 1,
            "No": 0
        }

        yesnomaybe_mapper = {
            "Yes": 1,
            "Maybe": 0.5,
            "No": 0
        }

        # df['Buyer Interest'] = df['Buyer Interest'].replace(scale_mapper)
        # df['Company Size'] = df['Company Size'].replace(scale_mapper_company_size)

        # df['Aviation'] = df['Aviation'].replace(yesno_mapper)
        # df['Automotive'] = df['Automotive'].replace(yesno_mapper)
        # df['Agriculture'] = df['Agriculture'].replace(yesno_mapper)
        # df['Banking/Finance/Insurance'] = df['Banking/Finance/Insurance'].replace(yesno_mapper)
        # df['Construction'] = df['Construction'].replace(yesno_mapper)
        # df['Chemicals/Life Sciences'] = df['Chemicals/Life Sciences'].replace(yesno_mapper)
        # df['Education'] = df['Education'].replace(yesno_mapper)
        # df['Energy/Utilities'] = df['Energy/Utilities'].replace(yesno_mapper)
        # df['Defense/Security'] = df['Defense/Security'].replace(yesno_mapper)
        # df['Government'] = df['Government'].replace(yesno_mapper)
        # df['Health'] = df['Health'].replace(yesno_mapper)
        # df['Hospitality'] = df['Hospitality'].replace(yesno_mapper)
        # df['Logistics'] = df['Logistics'].replace(yesno_mapper)
        # df['Manufacturing'] = df['Manufacturing'].replace(yesno_mapper)
        # df['Media/Entertainment'] = df['Media/Entertainment'].replace(yesno_mapper)
        # df['Public Sector'] = df['Public Sector'].replace(yesno_mapper)
        # df['Retail and eCommerce'] = df['Retail and eCommerce'].replace(yesno_mapper)
        # df['Transport'] = df['Transport'].replace(yesno_mapper)
        # df['Travel'] = df['Travel'].replace(yesno_mapper)
        # df['Telecommunications'] = df['Telecommunications'].replace(yesno_mapper)
        # df['Personal Contact'] = df['Personal Contact'].replace(yesnomaybe_mapper)
        # df['Conference'] = df['Conference'].replace(yesnomaybe_mapper)
        # df['Company Size'] = df['Company Size'].replace(yesno_mapper)
        # df['North America'] = df['North America'].replace(yesno_mapper)
        # df['LATAM'] = df['LATAM'].replace(yesno_mapper)
        # df['UK'] = df['UK'].replace(yesno_mapper)
        # df['Europe'] = df['Europe'].replace(yesno_mapper)
        # df['MEA'] = df['MEA'].replace(yesno_mapper)
        # df['APAC'] = df['APAC'].replace(yesno_mapper)
        # df['Lead became Opportunity'] = df['Lead became Opportunity'].replace(yesno_mapper)

        df_trial['Buyer Interest'] = df_trial['Buyer Interest'].replace(scale_mapper)
        df_trial['Company Size'] = df_trial['Company Size'].replace(scale_mapper_company_size)

        df_trial['Aviation'] = df_trial['Aviation'].replace(yesno_mapper)
        df_trial['Automotive'] = df_trial['Automotive'].replace(yesno_mapper)
        df_trial['Agriculture'] = df_trial['Agriculture'].replace(yesno_mapper)
        df_trial['Banking/Finance/Insurance'] = df_trial['Banking/Finance/Insurance'].replace(yesno_mapper)
        df_trial['Construction'] = df_trial['Construction'].replace(yesno_mapper)
        df_trial['Chemicals/Life Sciences'] = df_trial['Chemicals/Life Sciences'].replace(yesno_mapper)
        df_trial['Education'] = df_trial['Education'].replace(yesno_mapper)
        df_trial['Energy/Utilities'] = df_trial['Energy/Utilities'].replace(yesno_mapper)
        df_trial['Defense/Security'] = df_trial['Defense/Security'].replace(yesno_mapper)
        df_trial['Government'] = df_trial['Government'].replace(yesno_mapper)
        df_trial['Health'] = df_trial['Health'].replace(yesno_mapper)
        df_trial['Hospitality'] = df_trial['Hospitality'].replace(yesno_mapper)
        df_trial['Logistics'] = df_trial['Logistics'].replace(yesno_mapper)
        df_trial['Manufacturing'] = df_trial['Manufacturing'].replace(yesno_mapper)
        df_trial['Media/Entertainment'] = df_trial['Media/Entertainment'].replace(yesno_mapper)
        df_trial['Public Sector'] = df_trial['Public Sector'].replace(yesno_mapper)
        df_trial['Retail and eCommerce'] = df_trial['Retail and eCommerce'].replace(yesno_mapper)
        df_trial['Transport'] = df_trial['Transport'].replace(yesno_mapper)
        df_trial['Travel'] = df_trial['Travel'].replace(yesno_mapper)
        df_trial['Telecommunications'] = df_trial['Telecommunications'].replace(yesno_mapper)
        df_trial['Personal Contact'] = df_trial['Personal Contact'].replace(yesnomaybe_mapper)
        df_trial['Conference'] = df_trial['Conference'].replace(yesnomaybe_mapper)
        df_trial['Company Size'] = df_trial['Company Size'].replace(yesno_mapper)
        df_trial['North America'] = df_trial['North America'].replace(yesno_mapper)
        df_trial['LATAM'] = df_trial['LATAM'].replace(yesno_mapper)
        df_trial['UK'] = df_trial['UK'].replace(yesno_mapper)
        df_trial['Europe'] = df_trial['Europe'].replace(yesno_mapper)
        df_trial['MEA'] = df_trial['MEA'].replace(yesno_mapper)
        df_trial['APAC'] = df_trial['APAC'].replace(yesno_mapper)
        df_trial['Lead became Opportunity'] = df_trial['Lead became Opportunity'].replace(yesno_mapper)

        pd.set_option('display.max_columns', 100)
        pd.set_option('display.max_rows', 100)
        # first one-hot encode the categorical columns with NaNs

        # df = pd.get_dummies(df, columns=['Industry Vertical', 'BICS Product Area'],
        #                     dummy_na=False,
        #                     drop_first=False)

        df_trial = pd.get_dummies(df_trial, columns=['Industry Vertical', 'BICS Product Area'],
                                  dummy_na=False,
                                  drop_first=False)

        # new_cols = [col for col in df.columns if col != 'Lead became Opportunity'] + ['Lead became Opportunity']
        # df = df[new_cols]

        new_cols = [col for col in df_trial.columns if col != 'Lead became Opportunity'] + ['Lead became Opportunity']
        df_trial = df_trial[new_cols]

        robust_scaler = RobustScaler()
        # df[['Annual Revenue',
        #     'Annual Employee Growth (%)']] = robust_scaler.fit_transform(df[['Annual Revenue',
        #                                                                      'Annual Employee Growth (%)']])

        df_trial[['Annual Revenue',
                  'Annual Employee Growth (%)']] = robust_scaler.fit_transform(df_trial[['Annual Revenue',
                                                                                         'Annual Employee Growth (%)']])

        minmax_scaler = MinMaxScaler()
        # df[['Annual Revenue',
        #     'Annual Employee Growth (%)']] = minmax_scaler.fit_transform(df[['Annual Revenue',
        #                                                                      'Annual Employee Growth (%)']])

        df_trial[['Annual Revenue',
                  'Annual Employee Growth (%)']] = minmax_scaler.fit_transform(df_trial[['Annual Revenue',
                                                                                         'Annual Employee Growth (%)']])

        # companynames = df['Company name']

        # labels = df['Lead became Opportunity']

        # df.drop('Company name', axis=1, inplace=True)
        # df.drop('Lead became Opportunity', axis=1, inplace=True)

        df_trial.drop('Company name', axis=1, inplace=True)
        df_trial.drop('Lead became Opportunity', axis=1, inplace=True)

        # dfsave = df

        # num_cols = len(df.columns)

        # len_df = len(df)
        # df_1 = df.iloc[:(len_df - 1), :]

        # df = df_1

        # companynames_1 = companynames.iloc[:(len_df - 1)]
        # companynames_trial = companynames.iloc[99:]
        # companynames = companynames_1

        # labels_1 = labels.iloc[:(len_df - 1)]
        # labels_trial = labels.iloc[(len_df-1):]
        # labels = labels_1

        return df_trial, df_trial_original
    

    df_trial, df_trial_original = data_preprocessing(file)

    with open('tmp/et_model.pkl', 'rb') as f:
        et_model = pickle.load(f)

    with open('tmp/rf_model.pkl', 'rb') as f:
        rf_model = pickle.load(f)

    with open('tmp/gb_model.pkl', 'rb') as f:
        gb_model = pickle.load(f)

    results = []
    for i in range(0, len(df_trial)):
        if (et_model.predict(df_trial)[i] + rf_model.predict(df_trial)[i] + gb_model.predict(df_trial)[i]) < 2:
            results.append(0)
        else:
            results.append(1)
    print(results)

    prediction_results = pd.DataFrame()
    prediction_results['Company Name'] = df_trial_original['Company name']
    prediction_results['Will be an opportunity'] = results
    trial_proba_scores = (et_model.predict_proba(df_trial)[:, 1] + rf_model.predict_proba(df_trial)[:,
                                                                   1] + gb_model.predict_proba(df_trial)[:, 1]) / 3

    prediction_results['Probability of opportunity'] = trial_proba_scores

    prediction_results

    # get current date and time
    current_datetime = str(datetime.datetime.now())

    char_datetime = re.sub(r'[\W_]', '', str(datetime.datetime.now()))
    char_datetime = char_datetime[:-6]
    prediction_results.to_csv(
        f'{ROOT_DIR}/tmp/prediction_results_' + char_datetime + '.csv')
    
    return prediction_results





