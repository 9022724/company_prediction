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



def regenerate_model(file):
    def adjusted_classes(y_scores, t):
        """
        This function adjusts class predictions based on the prediction threshold (t).
        Will only work for binary classification problems.
        """
        return [1 if y >= t else 0 for y in y_scores]
    

    def optimise_threshold(model, importance_factor, dataset):
        y_scores = model.predict_proba(dataset)[:, 1]
        j = [x * 0.05 for x in range(0, 20)]
        optimal_score = 0
        optimal_threshold = 0
        halfway_recall_score = 0
        for thresh in j:
            y_pred_adj = adjusted_classes(y_scores, thresh)
            if len(dataset) == len(labels):
                neg, pos = confusion_matrix(labels, y_pred_adj)
            else:
                neg, pos = confusion_matrix(y_test, y_pred_adj)
            true_negatives = neg[0]
            false_positives = neg[1]
            false_negatives = pos[0]
            true_positives = pos[1]
            # true_recall_score = (true_positives * importance_factor) + (true_negatives)
            true_recall_score = true_positives + true_negatives - (
                    false_negatives * importance_factor) - false_positives
            # Recall = TruePositives / (TruePositives + FalseNegatives)
            if optimal_score == 0:
                optimal_score = true_recall_score
                optimal_predictions = y_pred_adj  # Risk here
            else:
                if true_recall_score >= optimal_score:
                    optimal_score = true_recall_score
                    optimal_threshold = thresh
                    optimal_predictions = y_pred_adj
            if thresh == 0.5:
                halfway_recall_score = true_recall_score
                halfway_predictions = y_pred_adj
        #        if len (dataset) == len (labels):
        #            print (thresh,true_recall_score)
        #            print(pd.DataFrame(confusion_matrix(labels, y_pred_adj),
        #                 columns=['pred_neg', 'pred_pos'], index=['neg', 'pos']))
        #        else:
        #            print (thresh,true_recall_score)
        #            print(pd.DataFrame(confusion_matrix(y_test, y_pred_adj),
        #                 columns=['pred_neg', 'pred_pos'], index=['neg', 'pos']))

        if halfway_recall_score == optimal_score:
            optimal_threshold = 0.5
            optimal_predictions = halfway_predictions
        if len(dataset) == len(labels):
            print(pd.DataFrame(confusion_matrix(labels, optimal_predictions),
                               columns=['pred_neg', 'pred_pos'], index=['neg', 'pos']))
        else:
            print(pd.DataFrame(confusion_matrix(y_test, optimal_predictions),
                               columns=['pred_neg', 'pred_pos'], index=['neg', 'pos']))
        print("")

        return (optimal_score, optimal_threshold, optimal_predictions, y_scores)
    

    def data_preprocessing(fil1):
        df = pd.read_csv(fil1)
        # df_trial = pd.read_csv(file2)

        df_original = df
        # df_trial_original = df_trial

        df.drop('Website', axis=1, inplace=True)
        df.drop('Comment', axis=1, inplace=True)
        df.drop('Salesforce Status', axis=1, inplace=True)
        df.tail()
        # df_trial.drop('Website', axis=1, inplace=True)
        # df_trial.drop('Comment', axis=1, inplace=True)
        # df_trial.drop('Salesforce Status', axis=1, inplace=True)

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

        df['Buyer Interest'] = df['Buyer Interest'].replace(scale_mapper)
        df['Company Size'] = df['Company Size'].replace(scale_mapper_company_size)

        df['Aviation'] = df['Aviation'].replace(yesno_mapper)
        df['Automotive'] = df['Automotive'].replace(yesno_mapper)
        df['Agriculture'] = df['Agriculture'].replace(yesno_mapper)
        df['Banking/Finance/Insurance'] = df['Banking/Finance/Insurance'].replace(yesno_mapper)
        df['Construction'] = df['Construction'].replace(yesno_mapper)
        df['Chemicals/Life Sciences'] = df['Chemicals/Life Sciences'].replace(yesno_mapper)
        df['Education'] = df['Education'].replace(yesno_mapper)
        df['Energy/Utilities'] = df['Energy/Utilities'].replace(yesno_mapper)
        df['Defense/Security'] = df['Defense/Security'].replace(yesno_mapper)
        df['Government'] = df['Government'].replace(yesno_mapper)
        df['Health'] = df['Health'].replace(yesno_mapper)
        df['Hospitality'] = df['Hospitality'].replace(yesno_mapper)
        df['Logistics'] = df['Logistics'].replace(yesno_mapper)
        df['Manufacturing'] = df['Manufacturing'].replace(yesno_mapper)
        df['Media/Entertainment'] = df['Media/Entertainment'].replace(yesno_mapper)
        df['Public Sector'] = df['Public Sector'].replace(yesno_mapper)
        df['Retail and eCommerce'] = df['Retail and eCommerce'].replace(yesno_mapper)
        df['Transport'] = df['Transport'].replace(yesno_mapper)
        df['Travel'] = df['Travel'].replace(yesno_mapper)
        df['Telecommunications'] = df['Telecommunications'].replace(yesno_mapper)
        df['Personal Contact'] = df['Personal Contact'].replace(yesnomaybe_mapper)
        df['Conference'] = df['Conference'].replace(yesnomaybe_mapper)
        df['Company Size'] = df['Company Size'].replace(yesno_mapper)
        df['North America'] = df['North America'].replace(yesno_mapper)
        df['LATAM'] = df['LATAM'].replace(yesno_mapper)
        df['UK'] = df['UK'].replace(yesno_mapper)
        df['Europe'] = df['Europe'].replace(yesno_mapper)
        df['MEA'] = df['MEA'].replace(yesno_mapper)
        df['APAC'] = df['APAC'].replace(yesno_mapper)
        df['Lead became Opportunity'] = df['Lead became Opportunity'].replace(yesno_mapper)

        pd.set_option('display.max_columns', 100)
        pd.set_option('display.max_rows', 100)
        # first one-hot encode the categorical columns with NaNs

        df = pd.get_dummies(df, columns=['Industry Vertical', 'BICS Product Area'],
                            dummy_na=False,
                            drop_first=False)


        new_cols = [col for col in df.columns if col != 'Lead became Opportunity'] + ['Lead became Opportunity']
        df = df[new_cols]


        robust_scaler = RobustScaler()
        df[['Annual Revenue',
            'Annual Employee Growth (%)']] = robust_scaler.fit_transform(df[['Annual Revenue',
                                                                             'Annual Employee Growth (%)']])

        minmax_scaler = MinMaxScaler()
        df[['Annual Revenue',
            'Annual Employee Growth (%)']] = minmax_scaler.fit_transform(df[['Annual Revenue',
                                                                             'Annual Employee Growth (%)']])

        companynames = df['Company name']

        labels = df['Lead became Opportunity']

        df.drop('Company name', axis=1, inplace=True)
        df.drop('Lead became Opportunity', axis=1, inplace=True)


        dfsave = df

        num_cols = len(df.columns)

        len_df = len(df)
        df_1 = df.iloc[:(len_df - 1), :]

        df = df_1

        companynames_1 = companynames.iloc[:(len_df - 1)]
        # companynames_trial = companynames.iloc[99:]
        companynames = companynames_1

        labels_1 = labels.iloc[:(len_df - 1)]
        # labels_trial = labels.iloc[(len_df-1):]
        labels = labels_1

        return labels, df, companynames, num_cols


    cutoff = 0
    probs = []

    def train_adaboost(df, labels, companynames):
        global cutoff, probs
        scaler = StandardScaler()
        n_pts = len(df)
        # Training examples
        n_train = int(0.7 * n_pts)

        # Divide into training and test sets with labels
        X_train, X_test, y_train, y_test = train_test_split(df, labels, train_size=n_train,
                                                            #                                                    random_state=0,
                                                            #                                                    stratify=labels,
                                                            random_state=0)

        scorers = {
            'precision_score': make_scorer(precision_score),
            'recall_score': make_scorer(recall_score),
            'accuracy_score': make_scorer(accuracy_score)
        }

        FN_factor = 15
        FN_factor_lower = 10

        # create the pipeline
        leadgen_pipe = Pipeline(steps=[('scaler', StandardScaler()), ('adaboost', AdaBoostClassifier())])

        # prepare a prameter grid
        param_grid = {
            'adaboost__learning_rate': [0.02, 0.05, 0.1, 0.3],
            'adaboost__algorithm': ['SAMME', 'SAMME.R'],
            'adaboost__random_state': [1]
        }

        print("Beginning AdaBoostClassifier", datetime.datetime.now())
        # search = GridSearchCV(leadgen_pipe, param_grid, n_jobs=-1, cv=5, refit=True)
        search = GridSearchCV(leadgen_pipe, param_grid, n_jobs=-1, cv=5, scoring=scorers, return_train_score=True,
                              refit='recall_score')
        search.fit(X_train, y_train)
        print("Finished AdaBoostClassifier", datetime.datetime.now())
        ab_model = search


        with open('tmp/ab_model.pkl', 'wb') as f:
            pickle.dump(ab_model, f)

        ab_prediction = ab_model.predict(X_test)

        print(f"Best AdaBoost score = {search.best_score_:0.3f}")
        print(f"Best parameters: {search.best_params_}")
        AB_best_params = search.best_params_
        AB_best_model = search.best_estimator_

        weights_array = []
        for i in range(0, num_cols):
            weights_array.append(
                [df.columns[i], search.best_estimator_.named_steps["adaboost"].feature_importances_[i]])
        ab_weights_df = pd.DataFrame(weights_array, columns=['Feature', 'Weight'])
        print(ab_weights_df.sort_values('Weight', ascending=False)[0:20])

        AB_optimal_score, AB_optimal_threshold, AB_optimal_predictions, AB_probabilities = optimise_threshold(ab_model,
                                                                                                              FN_factor,
                                                                                                              X_test)
        print(AB_optimal_score, AB_optimal_threshold, AB_optimal_predictions, AB_probabilities)

        AB_optimal_score, AB_optimal_threshold, AB_optimal_predictions_, AB_probabilities_ = optimise_threshold(
            ab_model,
            FN_factor,
            df)
        cutoff = AB_optimal_threshold

        count = 0
        Opportunities = []
        for i in AB_optimal_predictions_:
            if i == 1:
                Opportunities.append(companynames.iloc[count])
            count = count + 1
        probs = AB_probabilities_
        return ab_weights_df, AB_best_model, AB_optimal_predictions, AB_optimal_score, AB_probabilities

    def train_decision_tree(df, labels, companynames):
        global cutoff, probs
        scaler = StandardScaler()
        n_pts = len(df)
        # Training examples
        n_train = int(0.7 * n_pts)

        # Divide into training and test sets with labels
        X_train, X_test, y_train, y_test = train_test_split(df, labels, train_size=n_train,
                                                            #                                                    random_state=0,
                                                            #                                                    stratify=labels,
                                                            random_state=0)

        scorers = {
            'precision_score': make_scorer(precision_score),
            'recall_score': make_scorer(recall_score),
            'accuracy_score': make_scorer(accuracy_score)
        }

        # create the pipeline
        leadgen_pipe = Pipeline(steps=[('scaler', StandardScaler()), ('decision', DecisionTreeClassifier())])

        # prepare a prameter grid
        param_grid = {
            'decision__criterion': ['gini', 'entropy'],
            'decision__splitter': ['best', 'random'],
            'decision__max_features': ['auto', 'sqrt', 'log2'],
            'decision__random_state': [2, 3]
        }

        print("Beginning DecisionTreeClassifier", datetime.datetime.now())
        search = GridSearchCV(leadgen_pipe, param_grid, n_jobs=-1, cv=5, scoring=scorers, return_train_score=True,
                              refit='recall_score')
        search.fit(X_train, y_train)
        print("Finished DecisionTreeClassifier", datetime.datetime.now())
        dt_model = search

        with open('tmp/dt_model.pkl', 'wb') as f:
            pickle.dump(dt_model, f)

        dt_prediction = dt_model.predict(X_test)

        print(f"Best DecisionTreeClassifier score = {search.best_score_:0.3f}")
        print(f"Best parameters: {search.best_params_}")
        DT_best_params = search.best_params_
        DT_best_model = search.best_estimator_

        weights_array = []
        for i in range(0, num_cols):
            weights_array.append(
                [df.columns[i], search.best_estimator_.named_steps["decision"].feature_importances_[i]])
        dt_weights_df = pd.DataFrame(weights_array, columns=['Feature', 'Weight'])
        print(dt_weights_df.sort_values('Weight', ascending=False)[0:20])

        DT_optimal_score, DT_optimal_threshold, DT_optimal_predictions, DT_probabilities = optimise_threshold(dt_model,
                                                                                                              FN_factor,
                                                                                                              X_test)
        print(DT_optimal_score, DT_optimal_threshold, DT_optimal_predictions, DT_probabilities)

        DT_optimal_score, DT_optimal_threshold, DT_optimal_predictions_, DT_probabilities_ = optimise_threshold(
            dt_model,
            FN_factor,
            df)
        # print (DT_optimal_score, DT_optimal_threshold, DT_optimal_predictions_, DT_probabilities)
        cutoff = cutoff + DT_optimal_threshold

        count = 0
        Opportunities = []
        print(len(DT_optimal_predictions_))
        for i in DT_optimal_predictions_:
            if i == 1:
                Opportunities.append(companynames.iloc[count])
            count = count + 1
        Opportunities
        probs = np.add(probs, DT_probabilities_)
        return dt_weights_df, DT_best_model, DT_optimal_predictions, DT_optimal_score, DT_probabilities

    def train_extra_trees(df, labels, companynames):
        global cutoff, probs
        scaler = StandardScaler()
        n_pts = len(df)
        # Training examples
        n_train = int(0.7 * n_pts)

        # Divide into training and test sets with labels
        X_train, X_test, y_train, y_test = train_test_split(df, labels, train_size=n_train,
                                                            #                                                    random_state=0,
                                                            #                                                    stratify=labels,
                                                            random_state=0)

        scorers = {
            'precision_score': make_scorer(precision_score),
            'recall_score': make_scorer(recall_score),
            'accuracy_score': make_scorer(accuracy_score)
        }

        # create the pipeline
        leadgen_pipe = Pipeline(steps=[('scaler', scaler), ('extratrees', ExtraTreesClassifier())])

        # prepare a prameter grid
        param_grid = {
            'extratrees__criterion': ['gini', 'entropy'],
            'extratrees__class_weight': ['balanced', 'balanced_subsample'],
            'extratrees__max_features': ['auto', 'sqrt', 'log2'],
            'extratrees__random_state': [2, 3]
        }

        print("Beginning ExtraTreesClassifier", datetime.datetime.now())
        # search = GridSearchCV(leadgen_pipe, param_grid, n_jobs=-1, cv=5, refit=True)
        search = GridSearchCV(leadgen_pipe, param_grid, n_jobs=-1, cv=5, scoring=scorers, return_train_score=True,
                              refit='recall_score')

        search.fit(X_train, y_train)
        print("Finished ExtraTreesClassifier", datetime.datetime.now())
        et_model = search

        with open('tmp/et_model.pkl', 'wb') as f:
            pickle.dump(et_model, f)

        et_prediction = et_model.predict(X_test)

        print(f"Best ExtraTreesClassifier score = {search.best_score_:0.3f}")
        print(f"Best parameters: {search.best_params_}")
        ET_best_params = search.best_params_
        ET_best_model = search.best_estimator_

        weights_array = []
        for i in range(0, num_cols):
            weights_array.append(
                [df.columns[i], search.best_estimator_.named_steps["extratrees"].feature_importances_[i]])
        et_weights_df = pd.DataFrame(weights_array, columns=['Feature', 'Weight'])
        print(et_weights_df.sort_values('Weight', ascending=False)[0:20])

        ET_optimal_score, ET_optimal_threshold, ET_optimal_predictions, ET_probabilities = optimise_threshold(et_model,
                                                                                                              FN_factor,
                                                                                                              X_test)
        # print (ET_optimal_score,ET_optimal_threshold, ET_optimal_predictions, ET_probabilities)
        cutoff = cutoff + ET_optimal_threshold

        ET_optimal_score, ET_optimal_threshold, ET_optimal_predictions_, ET_probabilities_ = optimise_threshold(
            et_model,
            FN_factor,
            df)
        # print (ET_optimal_score,ET_optimal_threshold, ET_full_predictions_, ET_probabilities)

        probs = np.add(ET_probabilities_, probs)
        count = 0
        Opportunities = []

        for i in ET_optimal_predictions_:
            if i == 1:
                Opportunities.append(companynames.iloc[count])
            count = count + 1

        return et_weights_df, ET_best_model, ET_optimal_predictions, ET_optimal_score, ET_probabilities, ET_probabilities_, ET_optimal_threshold, et_model

    def train_gradiant_boosting(df, labels, companynames):
        global cutoff, probs
        scaler = StandardScaler()
        n_pts = len(df)
        # Training examples
        n_train = int(0.7 * n_pts)

        # Divide into training and test sets with labels
        X_train, X_test, y_train, y_test = train_test_split(df, labels, train_size=n_train,
                                                            #                                                    random_state=0,
                                                            #                                                    stratify=labels,
                                                            random_state=0)

        scorers = {
            'precision_score': make_scorer(precision_score),
            'recall_score': make_scorer(recall_score),
            'accuracy_score': make_scorer(accuracy_score)
        }

        # create the pipeline
        leadgen_pipe = Pipeline(steps=[('scaler', scaler), ('gradboost', GradientBoostingClassifier())])

        # prepare a prameter grid
        param_grid = {
            'gradboost__loss': ['exponential', 'deviance'],
            'gradboost__learning_rate': [0.05, 0.1, 0.2],
            'gradboost__criterion': ['friedman_mse', 'squared_error', 'mse'],
            'gradboost__max_features': ['auto', 'sqrt', 'log2'],
            'gradboost__random_state': [3]
        }

        print("Beginning GradientBoostingClassifier", datetime.datetime.now())
        # search = GridSearchCV(leadgen_pipe, param_grid, n_jobs=-1, cv=5, refit=True)
        search = GridSearchCV(leadgen_pipe, param_grid, n_jobs=-1, cv=5, scoring=scorers, return_train_score=True,
                              refit='recall_score')
        search.fit(X_train, y_train)
        print("Finished GradientBoostingClassifier", datetime.datetime.now())
        gb_model = search

        with open('tmp/gb_model.pkl', 'wb') as f:
            pickle.dump(gb_model, f)

        gb_prediction = gb_model.predict(X_test)

        print(f"Best GradientBoostingClassifier score = {search.best_score_:0.3f}")
        print(f"Best parameters: {search.best_params_}")
        GB_best_params = search.best_params_
        GB_best_model = search.best_estimator_

        GB_optimal_score, GB_optimal_threshold, GB_optimal_predictions, GB_probabilities = optimise_threshold(gb_model,
                                                                                                              FN_factor,
                                                                                                              X_test)
        # print (GB_optimal_score,GB_optimal_threshold, GB_optimal_predictions, GB_probabilities)

        weights_array = []
        for i in range(0, num_cols):
            weights_array.append(
                [df.columns[i], search.best_estimator_.named_steps["gradboost"].feature_importances_[i]])
        gb_weights_df = pd.DataFrame(weights_array, columns=['Feature', 'Weight'])
        print(gb_weights_df.sort_values('Weight', ascending=False)[0:20])

        gb_weights_df.sort_values('Weight', ascending=False).to_csv(
            rf'{ROOT_DIR}/tmp/Feature Importance' + str(datetime.datetime.now())[
                                                                                          20:], index=False)

        GB_optimal_score, GB_optimal_threshold, GB_optimal_predictions_, GB_probabilities_ = optimise_threshold(
            gb_model,
            FN_factor,
            df)
        # print (GB_optimal_score,GB_optimal_threshold, GB_optimal_predictions_, GB_probabilities)

        cutoff = cutoff + GB_optimal_threshold
        count = 0
        Opportunities = []

        for i in GB_optimal_predictions_:
            if i == 1:
                Opportunities.append(companynames.iloc[count])
            count = count + 1
        Opportunities

        probs = np.add(probs, GB_probabilities_)
        print(GB_optimal_score, GB_optimal_threshold)
        return gb_weights_df, GB_best_model, GB_optimal_predictions, GB_optimal_score, GB_probabilities, GB_probabilities_, GB_optimal_threshold, gb_model

    def train_random_forest(df, labels, companynames):
        global cutoff, probs
        scaler = StandardScaler()
        n_pts = len(df)
        # Training examples
        n_train = int(0.7 * n_pts)

        # Divide into training and test sets with labels
        X_train, X_test, y_train, y_test = train_test_split(df, labels, train_size=n_train,
                                                            #                                                    random_state=0,
                                                            #                                                    stratify=labels,
                                                            random_state=0)

        # create the pipeline
        leadgen_pipe = Pipeline(steps=[('scaler', scaler), ('rf', RandomForestClassifier())])

        # prepare a prameter grid
        param_grid = {
            'rf__n_estimators': [1000, 2000, 5000],
            'rf__max_depth': [14, 16, 18],
            'rf__random_state': [2]
        }

        print("Beginning Random Forest", datetime.datetime.now())
        search = GridSearchCV(leadgen_pipe, param_grid, n_jobs=-1, cv=5, refit=True)
        search.fit(X_train, y_train)
        print("Finished Random Forest", datetime.datetime.now())
        rf_model = search

        with open('tmp/rf_model.pkl', 'wb') as f:
            pickle.dump(rf_model, f)

        rf_prediction = rf_model.predict(X_test)

        print(f"Best Random Forest score = {search.best_score_:0.3f}")
        print(f"Best parameters: {search.best_params_}")
        RF_best_params = search.best_params_
        RF_best_model = search.best_estimator_

        RF_optimal_score, RF_optimal_threshold, RF_optimal_predictions, RF_probabilities = optimise_threshold(rf_model,
                                                                                                              FN_factor,
                                                                                                              X_test)
        # print (RF_optimal_score,RF_optimal_threshold, RF_optimal_predictions, RF_probabilities)

        weights_array = []
        for i in range(0, num_cols):
            weights_array.append([df.columns[i], search.best_estimator_.named_steps["rf"].feature_importances_[i]])
        rf_weights_df = pd.DataFrame(weights_array, columns=['Feature', 'Weight'])
        print(rf_weights_df.sort_values('Weight', ascending=False)[0:20])

        rf_weights_df.sort_values('Weight', ascending=False).to_csv(
            rf'{ROOT_DIR}/tmp/Feature Importance' + str(datetime.datetime.now())[
                                                                                          20:], index=False)

        RF_optimal_score, RF_optimal_threshold, RF_optimal_predictions_, RF_probabilities_ = optimise_threshold(
            rf_model,
            FN_factor,
            df)
        # print (RF_optimal_score,RF_optimal_threshold, RF_optimal_predictions_, RF_probabilities)

        cutoff = cutoff + RF_optimal_threshold
        count = 0
        Opportunities = []

        for i in RF_optimal_predictions_:
            if i == 1:
                Opportunities.append(companynames.iloc[count])
            count = count + 1
        Opportunities

        probs = np.add(probs, RF_probabilities_)
        return rf_weights_df, RF_best_model, RF_optimal_predictions, RF_optimal_score, RF_probabilities, RF_probabilities_, RF_optimal_threshold, rf_model

    def train_logistic_regression(df, labels, companynames):
        global cutoff, probs
        scaler = StandardScaler()
        n_pts = len(df)
        # Training examples
        n_train = int(0.7 * n_pts)

        # Divide into training and test sets with labels
        X_train, X_test, y_train, y_test = train_test_split(df, labels, train_size=n_train,
                                                            #                                                    random_state=0,
                                                            #                                                    stratify=labels,
                                                            random_state=0)

        scorers = {
            'precision_score': make_scorer(precision_score),
            'recall_score': make_scorer(recall_score),
            'accuracy_score': make_scorer(accuracy_score)
        }

        logistic = LogisticRegression(max_iter=10000, tol=0.1)

        # create the pipeline
        leadgen_pipe = Pipeline(steps=[('scaler', StandardScaler()), ('lr', logistic)])

        # prepare a prameter grid
        param_grid = {
            'lr__C': [0.01, 0.02, 0.05, 0.1],
            'lr__solver': ['lbfgs', 'liblinear', 'sag', 'saga']
        }

        print("Beginning Logistic Regression", datetime.datetime.now())
        # search = GridSearchCV(leadgen_pipe, param_grid, n_jobs=-1, cv=5, refit=True)
        search = GridSearchCV(leadgen_pipe, param_grid, n_jobs=-1, cv=5, scoring=scorers, return_train_score=True,
                              refit='recall_score')
        search.fit(X_train, y_train)
        print("Finished Logistic Regression", datetime.datetime.now())
        lr_model = search

        with open('tmp/lr_model.pkl', 'wb') as f:
            pickle.dump(lr_model, f)

        lr_prediction = lr_model.predict(X_test)

        print(f"Best Logistic Regression score = {search.best_score_:0.3f}")
        print(f"Best parameters: {search.best_params_}")
        LR_best_params = search.best_params_
        LR_best_model = search.best_estimator_

        LR_optimal_score, LR_optimal_threshold, LR_optimal_predictions, LR_probabilities = optimise_threshold(lr_model,
                                                                                                              FN_factor_lower,
                                                                                                              X_test)
        # print (LR_optimal_score,LR_optimal_threshold, LR_optimal_predictions, LR_probabilities)

        weights_array = []
        for i in range(0, num_cols):
            weights_array.append([df.columns[i], search.best_estimator_.named_steps["lr"].coef_[0][i]])
        lr_weights_df = pd.DataFrame(weights_array, columns=['Feature', 'Weight'])
        print(lr_weights_df.sort_values('Weight', ascending=False)[0:20])

        lr_weights_df.sort_values('Weight', ascending=False).to_csv(
            rf'{ROOT_DIR}/tmp/Feature Importance' + str(datetime.datetime.now())[
                                                                                          20:], index=False)

        LR_optimal_score, LR_optimal_threshold, LR_optimal_predictions_, LR_probabilities_ = optimise_threshold(
            lr_model,
            FN_factor,
            df)
        # print (LR_optimal_score,LR_optimal_threshold, LR_optimal_predictions_, LR_probabilities)
        cutoff = cutoff + LR_optimal_threshold

        count = 0
        Opportunities = []
        # print (len(LR_optimal_predictions_))
        for i in LR_optimal_predictions_:
            if i == 1:
                Opportunities.append(companynames.iloc[count])
            count = count + 1
        # Opportunities
        probs = np.add(probs, LR_probabilities_)
        return lr_weights_df, LR_best_model, LR_optimal_predictions, LR_optimal_score, LR_probabilities

    def train_multilayer_perceptron(df, labels, companynames):
        global cutoff, probs
        scaler = StandardScaler()
        n_pts = len(df)
        # Training examples
        n_train = int(0.7 * n_pts)

        # Divide into training and test sets with labels
        X_train, X_test, y_train, y_test = train_test_split(df, labels, train_size=n_train,
                                                            #                                                    random_state=0,
                                                            #                                                    stratify=labels,
                                                            random_state=0)

        scorers = {
            'precision_score': make_scorer(precision_score),
            'recall_score': make_scorer(recall_score),
            'accuracy_score': make_scorer(accuracy_score)
        }

        leadgen_pipe = Pipeline(
            steps=[('scaler', StandardScaler()), ('mlpc', neural_network.MLPClassifier(random_state=1))])

        param_grid = {
            'mlpc__max_iter': [1000, 2000],
            'mlpc__solver': ['lbfgs', 'sgd', 'adam'],
            'mlpc__learning_rate': ['constant', 'invscaling', 'adaptive'],
            'mlpc__activation': ['identity', 'logistic', 'tanh', 'relu'],
            'mlpc__random_state': [3],
            'mlpc__shuffle': [True]}

        print("Beginning MLPC", datetime.datetime.now())
        # search = GridSearchCV(leadgen_pipe, param_grid, n_jobs=-1, cv=5, refit=True)
        search = GridSearchCV(leadgen_pipe, param_grid, n_jobs=-1, cv=5, scoring=scorers, return_train_score=True,
                              refit='recall_score')
        search.fit(X_train, y_train)
        print("Finished MLPC", datetime.datetime.now())
        mlpc_model = search

        with open('tmp/mlpc_model.pkl', 'wb') as f:
            pickle.dump(mlpc_model, f)

        mlpc_prediction = mlpc_model.predict(X_test)
        mlpc_prob = mlpc_model.predict_proba(X_test)

        print("Best MLPC score = %0.3f:" % search.best_score_)
        print("Best parameters: ", search.best_params_)

        # store the best params and best model for later use
        MLPC_best_params = search.best_params_
        MLPC_best_model = search.best_estimator_

        MLPC_optimal_score, MLPC_optimal_threshold, MLPC_optimal_predictions, MLPC_probabilities = optimise_threshold(
            mlpc_model, FN_factor, X_test)
        # print (MLPC_optimal_score,MLPC_optimal_threshold, MLPC_optimal_predictions, MLPC_probabilities)

        #     weights_array = []
        #     for i in range (0,num_cols):
        #         weights_array.append ([df.columns[i],search.best_estimator_.named_steps["mlpc"].coefs_[0][i]])
        #     mlpc_weights_df = pd.DataFrame(weights_array, columns=['Feature', 'Weight'])
        #     #print(mlpc_weights_df['Weight'][0])
        #     print (mlpc_weights_df['Weight'].sum().sort())

        #     mlpc_weights_df.sort_values('Weight',ascending=False).to_csv (rf'{ROOT_DIR}/Feature Importance'+str(datetime.datetime.now())[20:], index = False)

        MLPC_optimal_score, MLPC_optimal_threshold, MLPC_optimal_predictions_, MLPC_probabilities_ = optimise_threshold(
            mlpc_model, FN_factor, df)
        # print (MLPC_optimal_score,MLPC_optimal_threshold, MLPC_optimal_predictions_, MLPC_probabilities)
        cutoff = cutoff + MLPC_optimal_threshold

        count = 0
        Opportunities = []
        # print (len(MLPC_optimal_predictions_))
        for i in MLPC_optimal_predictions_:
            if i == 1:
                Opportunities.append(companynames.iloc[count])
            count = count + 1
        Opportunities
        probs = np.add(probs, MLPC_probabilities_)
        return MLPC_best_model, MLPC_optimal_predictions, MLPC_optimal_score, MLPC_probabilities

    def train_support_vector_machine(df, labels, companynames):
        global cutoff, probs
        scaler = StandardScaler()
        n_pts = len(df)
        # Training examples
        n_train = int(0.7 * n_pts)

        # Divide into training and test sets with labels
        X_train, X_test, y_train, y_test = train_test_split(df, labels, train_size=n_train,
                                                            #                                                    random_state=0,
                                                            #                                                    stratify=labels,
                                                            random_state=0)

        # create the pipeline
        leadgen_pipe = Pipeline(steps=[('scaler', scaler), ('svm', svm.SVC(probability=True))])

        # prepare a parameter grid
        param_grid = {
            'svm__C': [0.05, 0.1, 0.3, 0.6, 1],
            'svm__kernel': ['linear', 'rbf', 'sigmoid', 'poly'],
            'svm__gamma': [40, 20, 10],
            'svm__random_state': [1]
        }

        print("Beginning Support Vector Machine", datetime.datetime.now())
        search = GridSearchCV(leadgen_pipe, param_grid, n_jobs=-1, cv=5, refit=True)
        search.fit(X_train, y_train)
        print("Finished Support Vector Machine", datetime.datetime.now())

        print(f"Best SVM score = {search.best_score_:0.3f}")
        print(f"Best parameters: {search.best_params_}")
        SVM_best_params = search.best_params_
        SVM_best_model = search.best_estimator_

        svm_model = search

        with open('tmp/svm_model.pkl', 'wb') as f:
            pickle.dump(svm_model, f)

        svm_prediction = svm_model.predict(X_test)
        accuracy = accuracy_score(y_test, svm_prediction)
        print("Accuracy: %.2f%%" % (accuracy * 100.0))

        SVM_optimal_score, SVM_optimal_threshold, SVM_optimal_predictions, SVM_probabilities = optimise_threshold(
            svm_model,
            FN_factor,
            X_test)

        weights_array = []
        for i in range(0, num_cols):
            weights_array.append([df.columns[i], search.best_estimator_.named_steps["svm"].coef_[0][i]])
        svm_weights_df = pd.DataFrame(weights_array, columns=['Feature', 'Weight'])
        print(svm_weights_df.sort_values('Weight', ascending=False)[0:20])

        svm_weights_df.sort_values('Weight', ascending=False).to_csv(
            rf'{ROOT_DIR}/tmp/Feature Importance' + str(datetime.datetime.now())[
                                                                                          20:], index=False)

        SVM_optimal_score, SVM_optimal_threshold, SVM_optimal_predictions_, SVM_probabilities_ = optimise_threshold(
            svm_model, FN_factor, df)
        print(SVM_optimal_score)
        cutoff = cutoff + SVM_optimal_threshold

        count = 0
        Opportunities = []
        # print (len(MLPC_optimal_predictions_))
        for i in SVM_optimal_predictions_:
            if i == 1:
                Opportunities.append(companynames.iloc[count])
            count = count + 1
        Opportunities
        probs = np.add(probs, SVM_probabilities_)

    def train_xgboost(df, labels, companynames):
        global cutoff, probs
        n_pts = len(df)
        # Training examples
        n_train = int(0.7 * n_pts)

        # Divide into training and test sets with labels
        X_train, X_test, y_train, y_test = train_test_split(df, labels, train_size=n_train,
                                                            #                                                    random_state=0,
                                                            #                                                    stratify=labels,
                                                            random_state=0)

        print("Beginning XGBoost", datetime.datetime.now())
        model = XGBClassifier()
        model.fit(X_train, y_train)
        print("Finished XGBoost", datetime.datetime.now())
        xgboost_model = model

        with open('tmp/xgboost_model.pkl', 'wb') as f:
            pickle.dump(xgboost_model, f)

        xgboost_prediction = xgboost_model.predict(X_test)
        accuracy = accuracy_score(y_test, xgboost_prediction)
        print("Accuracy: %.2f%%" % (accuracy * 100.0))

        XGB_optimal_score, XGB_optimal_threshold, XGB_optimal_predictions, XGB_probabilities = optimise_threshold(
            xgboost_model, FN_factor, X_test)
        # print (XGB_optimal_score,XGB_optimal_threshold, XGB_optimal_predictions, XGB_probabilities)

        weights_array = []
        for i in range(0, num_cols):
            weights_array.append([df.columns[i], xgboost_model.feature_importances_[i]])
        xgb_weights_df = pd.DataFrame(weights_array, columns=['Feature', 'Weight'])
        print(xgb_weights_df.sort_values('Weight', ascending=False)[0:20])

        XGB_optimal_score, XGB_optimal_threshold, XGB_optimal_predictions_, XGB_probabilities_ = optimise_threshold(
            xgboost_model, FN_factor, df)
        print(XGB_optimal_score)
        cutoff = cutoff + XGB_optimal_threshold

        count = 0
        Opportunities = []
        # print (len(XGB_optimal_predictions_))
        for i in XGB_optimal_predictions_:
            if i == 1:
                Opportunities.append(companynames.iloc[count])
            count = count + 1
        # Opportunities
        probs = np.add(probs, XGB_probabilities_)
        return xgb_weights_df, xgboost_model, XGB_optimal_predictions, XGB_optimal_score, XGB_probabilities
    

    def prepare_final_weights(et_weights_df, rf_weights_df, ab_weights_df, gb_weights_df, lr_weights_df, dt_weights_df,
                              xgb_weights_df):
        current_datetime = str(datetime.datetime.now())
        char_datetime = re.sub(r'[\W_]', '', str(datetime.datetime.now()))
        char_datetime = char_datetime[:-6]

        # weights_array = [et_weights_df, rf_weights_df, ab_weights_df]
        weights_array = [et_weights_df, rf_weights_df, ab_weights_df, gb_weights_df, lr_weights_df, dt_weights_df,
                         xgb_weights_df]
        final_weights = pd.concat(weights_array)
        weights_result = final_weights.groupby('Feature').apply(lambda x: x['Weight'].sum()).reset_index(name='Value')
        weights_result.sort_values('Value', ascending=False).to_csv(
            f'{ROOT_DIR}/tmp/Most important Features_' + char_datetime + '.csv')
        weights_result.sort_values('Value', ascending=False)[0:15]

    def evaluate_model(X_test, y_test, model, predictions):
        mean_fpr = np.linspace(start=0, stop=1, num=100)
        # compute probabilistic predictiond for the evaluation set
        _probabilities = model.predict_proba(X_test)[:, 1]

        # compute exact predictiond for the evaluation set
        _predicted_values = predictions

        # compute accuracy
        _accuracy = accuracy_score(y_test, _predicted_values)

        # compute precision, recall and f1 score for class 1
        _precision, _recall, _f1_score, _ = precision_recall_fscore_support(y_test, _predicted_values, labels=[1])

        # compute fpr and tpr values for various thresholds
        # by comparing the true target values to the predicted probabilities for class 1
        _fpr, _tpr, _ = roc_curve(y_test, _probabilities)

        # compute true positive rates for the values in the array mean_fpr
        _tpr_transformed = np.array([np.interp(mean_fpr, _fpr, _tpr)])

        # compute the area under the curve
        _auc = auc(_fpr, _tpr)

        return _accuracy, _precision[0], _recall[0], _f1_score[0], _tpr_transformed, _auc
    

    def evaluate_all_models(df, labels):
        n_pts = len(df)
        # Training examples
        n_train = int(0.7 * n_pts)

        # Divide into training and test sets with labels
        X_train, X_test, y_train, y_test = train_test_split(df, labels, train_size=n_train,
                                                            #                                                    random_state=0,
                                                            #                                                    stratify=labels,
                                                            random_state=0)
        print("a")
        AB_accuracy, AB_precision, AB_recall, AB_f1_score, AB_tpr, AB_auc = evaluate_model(X_test, y_test,
                                                                                           AB_best_model,
                                                                                           AB_optimal_predictions)
        print("b")
        DT_accuracy, DT_precision, DT_recall, DT_f1_score, DT_tpr, DT_auc = evaluate_model(X_test, y_test,
                                                                                           DT_best_model,
                                                                                           DT_optimal_predictions)
        print("c")
        ET_accuracy, ET_precision, ET_recall, ET_f1_score, ET_tpr, ET_auc = evaluate_model(X_test, y_test,
                                                                                           ET_best_model,
                                                                                           ET_optimal_predictions)
        print("d")
        GB_accuracy, GB_precision, GB_recall, GB_f1_score, GB_tpr, GB_auc = evaluate_model(X_test, y_test,
                                                                                           GB_best_model,
                                                                                           GB_optimal_predictions)
        print("e")
        RF_accuracy, RF_precision, RF_recall, RF_f1_score, RF_tpr, RF_auc = evaluate_model(X_test, y_test,
                                                                                           RF_best_model,
                                                                                           RF_optimal_predictions)
        print('f')
        LR_accuracy, LR_precision, LR_recall, LR_f1_score, LR_tpr, LR_auc = evaluate_model(X_test, y_test,
                                                                                           LR_best_model,
                                                                                           LR_optimal_predictions)
        print('g')
        MLPC_accuracy, MLPC_precision, MLPC_recall, MLPC_f1_score, MLPC_tpr, MLPC_auc = evaluate_model(X_test, y_test,
                                                                                                       MLPC_best_model,
                                                                                                       MLPC_optimal_predictions)
        print("h")
        XGB_accuracy, XGB_precision, XGB_recall, XGB_f1_score, XGB_tpr, XGB_auc = evaluate_model(X_test, y_test,
                                                                                                 xgboost_model,
                                                                                                 XGB_optimal_predictions)
        print("i")

        plt.figsize = (10, 8)
        plt.rcParams["figure.figsize"] = (13, 10)

        # SVM_metrics = np.array([SVM_accuracy, SVM_precision, SVM_recall, SVM_f1_score])
        AB_metrics = np.array([AB_accuracy, AB_precision, AB_recall, AB_f1_score])
        DT_metrics = np.array([DT_accuracy, DT_precision, DT_recall, DT_f1_score])
        ET_metrics = np.array([ET_accuracy, ET_precision, ET_recall, ET_f1_score])
        GB_metrics = np.array([GB_accuracy, GB_precision, GB_recall, GB_f1_score])
        RF_metrics = np.array([RF_accuracy, RF_precision, RF_recall, RF_f1_score])
        LR_metrics = np.array([LR_accuracy, LR_precision, LR_recall, LR_f1_score])
        MLPC_metrics = np.array([MLPC_accuracy, MLPC_precision, MLPC_recall, MLPC_f1_score])
        XGB_metrics = np.array([XGB_accuracy, XGB_precision, XGB_recall, XGB_f1_score])
        print("j")
        print("")
        plt.fontsize = 14
        index = ['accuracy', 'precision', 'recall', 'F1-score']
        df_metrics = pd.DataFrame(
            {'XGBoost': XGB_metrics, 'Ada Boost': AB_metrics, 'Decision Tree': DT_metrics, 'Extra Tress': ET_metrics,
             'Gradient Boosting': GB_metrics, 'Random Forest': RF_metrics, 'Logistic Regression': LR_metrics,
             'Multilayer Perceptron': MLPC_metrics}, index=index)
        df_metrics.plot.bar(rot=0, fontsize=14)
        print("k")
        plt.fontsize = 14
        plt.rcParams["figure.figsize"] = (13, 10)
        plt.legend(loc="lower left", fontsize=14)
        # plt.xlabel(fontsize=14)
        plt.ylim([0.1, 1])
        plt.title("Opportunity Analysis Accuracy, Precision, Recall & F1 Scores", fontsize=20)
        current_datetime = str(datetime.datetime.now())
        char_datetime = re.sub(r'[\W_]', '', str(datetime.datetime.now()))
        char_datetime = char_datetime[:-6]
        plt.savefig(f'{ROOT_DIR}/tmp/Accuracy_Precision_Recall_F1_' + char_datetime + '.pdf')
        # #     plt.show()

        mean_fpr = np.linspace(start=0, stop=1, num=100)
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=0.8)
        # plt.plot(mean_fpr, SVM_tpr[0,:], lw=2, color='grey', label='Support Vector Machine (AUC = %0.2f)' % (SVM_auc), alpha=0.8)
        plt.plot(mean_fpr, AB_tpr[0, :], lw=2, color='blue', label='Ada Boost (AUC = %0.2f)' % (AB_auc), alpha=0.8)
        plt.plot(mean_fpr, DT_tpr[0, :], lw=2, color='red', label='Decision Tree (AUC = %0.2f)' % (DT_auc), alpha=0.8)
        plt.plot(mean_fpr, ET_tpr[0, :], lw=2, color='orange', label='Extra Trees (AUC = %0.2f)' % (ET_auc), alpha=0.8)
        plt.plot(mean_fpr, GB_tpr[0, :], lw=2, color='violet', label='Gradient Boosting (AUC = %0.2f)' % (GB_auc),
                 alpha=0.8)
        plt.plot(mean_fpr, LR_tpr[0, :], lw=2, color='green', label='Logistic Regression (AUC = %0.2f)' % (LR_auc),
                 alpha=0.8)
        plt.plot(mean_fpr, MLPC_tpr[0, :], lw=2, color='purple',
                 label='Multi-Layer Perceptron (AUC = %0.2f)' % (MLPC_auc),
                 alpha=0.8)
        plt.plot(mean_fpr, RF_tpr[0, :], lw=2, color='black', label='Random Forest (AUC = %0.2f)' % (RF_auc), alpha=0.8)
        plt.plot(mean_fpr, XGB_tpr[0, :], lw=2, color='grey', label='XGBoost (AUC = %0.2f)' % (XGB_auc), alpha=0.8)
        plt.fontsize = 14
        plt.xlabel('False Positive Rate', fontsize=20)
        plt.ylabel('True Positive Rate', fontsize=20)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.title('Lead Generation: ROC Curves for Multiple Classifiers', fontsize=20)
        plt.rcParams["figure.figsize"] = (12, 9)
        plt.legend(loc="lower right", fontsize=14)
        print("ll")
        current_datetime = str(datetime.datetime.now())
        char_datetime = re.sub(r'[\W_]', '', str(datetime.datetime.now()))
        char_datetime = char_datetime[:-6]
        plt.savefig(f'{ROOT_DIR}/tmp/ROC_Curve_' + char_datetime + '.pdf')



    def ensumble_learning(AB_optimal_score, DT_optimal_score, ET_optimal_score, GB_optimal_score, LR_optimal_score,
                        MLPC_optimal_score, RF_optimal_score, XGB_optimal_score, AB_optimal_predictions,
                        DT_optimal_predictions, ET_optimal_predictions, GB_optimal_predictions,
                        LR_optimal_predictions,
                        MLPC_optimal_predictions, RF_optimal_predictions, XGB_optimal_predictions, AB_probabilities,
                        DT_probabilities, ET_probabilities, GB_probabilities, LR_probabilities, MLPC_probabilities,
                        RF_probabilities, XGB_probabilities):
        final_pred = np.array([])
        ensemble_probs = np.array([])

        model_names = ["AB_optimal_predictions[i]", "DT_optimal_predictions[i]", "ET_optimal_predictions[i]",
                       "GB_optimal_predictions[i]", "LR_optimal_predictions[i]", "MLPC_optimal_predictions[i]",
                       "RF_optimal_predictions[i]", "XGB_optimal_predictions[i]"]
        model_scores = [AB_optimal_score, DT_optimal_score, ET_optimal_score, GB_optimal_score, LR_optimal_score,
                        MLPC_optimal_score, RF_optimal_score, XGB_optimal_score]
        print(model_scores)
        # print (model_names[0][:2])
        scores_df = pd.DataFrame()
        scores_df['Models'] = model_names
        scores_df['Scores'] = model_scores
        scores_df['Predictions'] = [AB_optimal_predictions, DT_optimal_predictions, ET_optimal_predictions,
                                    GB_optimal_predictions,
                                    LR_optimal_predictions, MLPC_optimal_predictions, RF_optimal_predictions,
                                    XGB_optimal_predictions]
        scores_df['Probabilities'] = [AB_probabilities, DT_probabilities, ET_probabilities, GB_probabilities,
                                      LR_probabilities, MLPC_probabilities, RF_probabilities, XGB_probabilities]
        scores_table = scores_df.sort_values('Scores', ascending=False).reset_index(drop=True)

        best_models = []

        for i in range(0, len(scores_df['Predictions'][0])):
            final_pred = np.append(final_pred, statistics.mode(
                [scores_df['Predictions'][0][i], scores_df['Predictions'][1][i], scores_df['Predictions'][2][i]]))
            

            # ensemble_probs = np.append(ensemble_probs, statistics.mean(
            #     [scores_df['Probabilities'][0][i], scores_df['Probabilities'][1][i], scores_df['Probabilities'][2][i]]))
        plt.figure(figsize=(10, 8))
        plt.title('Opportunity Confusion Matrix - Ensemble Learning', fontsize=20)
        cm = metrics.confusion_matrix(y_test, final_pred)
        sns.heatmap(cm, annot=True, fmt=".1f", linewidths=.6)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.ylabel('Actual Label', fontsize=14)
        plt.xlabel('Predicted Label', fontsize=14)

        current_datetime = str(datetime.datetime.now())
        char_datetime = re.sub(r'[\W_]', '', str(datetime.datetime.now()))
        char_datetime = char_datetime[:-6]
        plt.savefig(
            f'{ROOT_DIR}/tmp/Ensemble Confusion Matrix_' + char_datetime + '.pdf')
    

    labels, df, companynames, num_cols = data_preprocessing(file)
    n_pts = len(df)
    # Training examples
    n_train = int(0.7 * n_pts)
    FN_factor = 15
    FN_factor_lower = 10


    X_train, X_test, y_train, y_test = train_test_split(df, labels, train_size=n_train, random_state=0)



    ab_weights_df, AB_best_model, AB_optimal_predictions, AB_optimal_score, AB_probabilities = train_adaboost(df,
                                                                                                              labels,
                                                                                                              companynames)

    dt_weights_df, DT_best_model, DT_optimal_predictions, DT_optimal_score, DT_probabilities = train_decision_tree(df,
                                                                                                                   labels,
                                                                                                                   companynames)

    et_weights_df, ET_best_model, ET_optimal_predictions, ET_optimal_score, ET_probabilities, ET_probabilities_, ET_optimal_threshold, et_model = train_extra_trees(
        df, labels, companynames)

    gb_weights_df, GB_best_model, GB_optimal_predictions, GB_optimal_score, GB_probabilities, GB_probabilities_, GB_optimal_threshold, gb_model = train_gradiant_boosting(
        df, labels, companynames)

    rf_weights_df, RF_best_model, RF_optimal_predictions, RF_optimal_score, RF_probabilities, RF_probabilities_, RF_optimal_threshold, rf_model = train_random_forest(
        df, labels, companynames)

    lr_weights_df, LR_best_model, LR_optimal_predictions, LR_optimal_score, LR_probabilities = train_logistic_regression(
        df,
        labels,
        companynames)

    MLPC_best_model, MLPC_optimal_predictions, MLPC_optimal_score, MLPC_probabilities = train_multilayer_perceptron(df,
                                                                                                                    labels,
                                                                                                                    companynames)

    train_support_vector_machine(df, labels, companynames)

    xgb_weights_df, xgboost_model, XGB_optimal_predictions, XGB_optimal_score, XGB_probabilities = train_xgboost(df,
                                                                                                                 labels,
                                                                                                                 companynames)
    

    print("calling prepare_final_weights")

    prepare_final_weights(et_weights_df, rf_weights_df, ab_weights_df, gb_weights_df, lr_weights_df, dt_weights_df,
                          xgb_weights_df)
    print("calling evaluate_all_models")
    evaluate_all_models(df, labels)
    print("1")
    will_be_opp_array = []
    dfresult = pd.DataFrame()
    print("2")
    dfresult['CompanyName'] = companynames
    # prob1 = np.add(AB_probabilities_,DT_probabilities_,LR_probabilities_)
    probs = np.add(GB_probabilities_, ET_probabilities_, RF_probabilities_)
    print("3")
    # probs = np.add(prob1,prob2)
    final_prediction = probs / 3 * 100
    # final_prediction = final_pred_long * 100
    # cutoff = AB_optimal_threshold + DT_optimal_threshold + ET_optimal_threshold + RF_optimal_threshold + LR_optimal_threshold + MLPC_optimal_threshold
    cutoff = ET_optimal_threshold + RF_optimal_threshold + GB_optimal_threshold
    finalcutoff = cutoff / 3 * 100
    dfresult['Opportunity Prediction %'] = final_prediction
    print("4")
    for i in range(0, len(final_prediction)):
        if final_prediction[i] > finalcutoff:
            will_be_opp_array.append(1)
        #            print (will_be_opp_array)
        else:
            will_be_opp_array.append(0)
    #            print (will_be_opp_array)
    # print (len(will_be_opp_array),len(dfresult))
    print("5")
    dfresult['Will be an Opportunity'] = will_be_opp_array
    # dfresult['Opportunity Prediction %']  = final_pred
    print("Opportunity Probability Cutoff is ", finalcutoff, "%")

    current_datetime = str(datetime.datetime.now())
    char_datetime = re.sub(r'[\W_]', '', str(datetime.datetime.now()))
    char_datetime = char_datetime[:-6]
    dfresult.sort_values('Opportunity Prediction %', ascending=False).to_csv(
        f'{ROOT_DIR}/tmp/Opportunity Prediction Percentages_' + char_datetime + '.csv')

    pd.set_option('display.max_rows', 200)
    dfresult.sort_values('Opportunity Prediction %', ascending=False)

    current_datetime = str(datetime.datetime.now())
    char_datetime = re.sub(r'[\W_]', '', str(datetime.datetime.now()))
    char_datetime = char_datetime[:-6]
    df.to_csv(f'{ROOT_DIR}/tmp/leadgendata_' + char_datetime + '.csv')


    ensumble_learning(AB_optimal_score, DT_optimal_score, ET_optimal_score, GB_optimal_score, LR_optimal_score,
                        MLPC_optimal_score, RF_optimal_score, XGB_optimal_score, AB_optimal_predictions,
                        DT_optimal_predictions, ET_optimal_predictions, GB_optimal_predictions,
                        LR_optimal_predictions,
                        MLPC_optimal_predictions, RF_optimal_predictions, XGB_optimal_predictions, AB_probabilities,
                        DT_probabilities, ET_probabilities, GB_probabilities, LR_probabilities, MLPC_probabilities,
                        RF_probabilities, XGB_probabilities)
    

    final_model = VotingClassifier(estimators=[('rf', rf_model), ('lr', et_model), ('et', et_model)], voting='hard')
    #final_model = VotingClassifier(estimators=[('rf', rf_model), ('lr', et_model), ('et', et_model)], voting='hard')
    final_model.fit(X_train,y_train)
    final_model.score(X_test,y_test)

    plt.figure(figsize=(10,8))
    plt.title('Opportunity Confusion Matrix - Voting', fontsize=20)
    cm = metrics.confusion_matrix(y_test, final_model.predict(X_test))
    sns.heatmap(cm, annot=True,  fmt=".1f",linewidths=.6)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylabel('Actual Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.savefig(f'{ROOT_DIR}/tmp/Voting_Confusion_Matrix_' + char_datetime + '.pdf')