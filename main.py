from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt

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
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_recall_fscore_support
import statistics
from sklearn import metrics

from leadgen_model import do_work

app = FastAPI()
templates = Jinja2Templates(directory="templates")

from fastapi import FastAPI, Request, UploadFile, File


@app.post("/", response_class=HTMLResponse)
def read_root(request: Request, file1: UploadFile = File(...), file2: UploadFile = File(...)):
    do_work(file1.file, file2.file)
    return templates.TemplateResponse("index.html", {"request": request, "message": "Request processed successfully"})


@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


if __name__ == '__main__':
    uvicorn.run("debug_server:app", host="0.0.0.0", port=80, reload=True)
