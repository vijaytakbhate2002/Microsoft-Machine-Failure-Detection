import pandas as pd
import numpy as np
import os
import logging
from src.best_model import BestModel
from src.Preprocess_data import DataProcessing
from src.Train import Train
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression

# Set up logging
logging.basicConfig(filename='model_log.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger().addHandler(logging.StreamHandler())

# Define algorithms
algos = {'rf': RandomForestClassifier(), 'dt': DecisionTreeClassifier(), 'logr': LogisticRegression(), 'svm': SVC()}

def pipeline_runner(data_path, target_col_name: str, deduction_by, test_size=0.2):
    """Returns: model, train_score, test_score, pr, re, f1
    Args:
        data_path: string path of dataset,
        deduction_by: ('test', 'train', 'cross') pick one from it
        test_size: test data size (default is 0.2, train data size is 0.8)
    """

    logging.debug("Pipeline runner started...")
    print("Pipeline runner started...")

    processor_class = DataProcessing(data_path)
    data = processor_class.preprocess_data()

    X, y = data.drop([target_col_name], axis='columns'), data[target_col_name]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    best_models = BestModel(X_train, X_test, y_train, y_test)
    best_models = best_models.deduction(deduction_by=deduction_by)  # dictionary
    best_algo = algos[list(best_models.keys())[0]]

    model, train_score, test_score, pr, re, f1 = Train(best_algo, X, y).train_model()

    logging.info("Model training completed.")
    print("Model training completed.")
    return model, train_score, test_score, pr, re, f1
