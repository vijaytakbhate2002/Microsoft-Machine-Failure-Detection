import streamlit as st
import pandas as pd
import numpy as np
import os
import logging
from src.best_model import BestModel
from src.Preprocess_data import DataProcessing
from src.Train import Train
from pipeline import pipeline_runner
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import logging
import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression
import joblib
import pickle

logging.basicConfig(filename='.\\model_log.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger().addHandler(logging.StreamHandler())

class ModelTrainer:
    """Class for training machine learning models.

    This class encapsulates functionalities for training various machine learning models
    based on user input and data. It handles model training and parameter setup.

    Attributes:
        FILE_PATH (str): The path to the data file used for training.
        TARGET (str): The target column name for prediction.
        flag (bool): A flag to track if a model has been trained.

    Methods:
        set_params: Sets up parameters required for model training.
        data_splitter: Splits the data into features and target variables.
        train: Orchestrates the training process based on user inputs.
    """

    FILE_PATH = "Data\\Complete dataframe.csv"
    TARGET = 'Error'
    flag = False

    def set_params(self,features:int)-> None:
        """Setting up of all parameters that are required for tunning our model"""

        with st.expander("Set Recommend best model"):
            deduction_by = st.selectbox("Validation by", options=('test', 'train', 'cross'))
            test_size = st.slider("Test size", 0.1, 0.5, value=0.2)

        # for Random Forest and Decision tree classifier
        with st.expander("Set Random Forest or Decision Tree"):
            n_estimators = st.slider("Number of trees (n_estimators)", min_value=1, max_value=100, value=10)
            max_depth = st.slider("Maximum depth of trees (max_depth) [Decision tree]", min_value=1, max_value=20, value=5)
            min_samples_split = st.slider("Minimum samples to split (min_samples_split [Decision tree]", min_value=2, max_value=10, value=2)
            min_samples_leaf = st.slider("Minimum samples per leaf (min_samples_leaf) [Decision tree]", min_value=1, max_value=4, value=1)
            max_features = st.slider("Max features to consider (max_features)", min_value=1, max_value=features, value=features // 2)
            bootstrap = st.checkbox("Use Bootstrap (instead of training on all the observations, each tree of RF is trained on a subset of the observations.)", value=True)

        # for logistic regression
        with st.expander("Set Logistic Regression"):
            C_value_log = st.slider("Inverse of regularization strength (C)", min_value=0.1, max_value=10.0, value=1.0)
            penalty_type = st.selectbox("Penalty type", ['l1', 'l2'])
            solver_type = st.selectbox("Solver type", ['lbfgs', 'liblinear', 'sag', 'saga'])
            if solver_type == 'lbfgs' and penalty_type == 'l1':
                penalty_type = 'l2'

        # for svm classifier
        with st.expander("Set Support Vector"):
            C_value_svm = st.slider("Regularization parameter (C)", min_value=0.1, max_value=10.0, value=1.0)
            degree_value = st.slider("Degree of the polynomial kernel function (degree)", min_value=1, max_value=10, value=3)
            kernel_type = st.selectbox("Kernel type", ['linear', 'poly', 'rbf', 'sigmoid'])

        variables_tuple = (n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, bootstrap,
        C_value_svm, degree_value, kernel_type,
        C_value_log, penalty_type, solver_type,
        deduction_by, test_size)
        return variables_tuple

    def data_splitter(self,path:str, target_col_name:str)->None:
        """Returns:tuple of dataframes (X,y)"""
        Data = DataProcessing(path=path).preprocess_data()
        return Data.drop(target_col_name,axis='columns'), Data[target_col_name]

    def download_model(self, model_bytes:ClassifierMixin)-> None:
        """Args: ClassifierMixin
           Return: None
           Used for downloading model_bytes (trained model) from streamlit server to users PC"""
        st.download_button(
        label="Download Model",
        data="classifier.pkl",
        file_name='model.pkl',
        mime='application/octet-stream'
        )

    def train(self)-> None:
        """handles all training process of model  with streamlit page"""
        X,y = self.data_splitter(path=self.FILE_PATH, target_col_name=self.TARGET)
        st.subheader("Set parameters", divider='rainbow')
        (n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, bootstrap,
        C_value_svm, degree_value, kernel_type,
        C_value_log, penalty_type, solver_type,
        deduction_by, test_size) = self.set_params(features=X.shape[1])

        st.subheader("Train your customized model", divider='rainbow')
        col1, col2, col3, col4, col5 = st.columns(5)
        
        button0 = col1.button('Recommend best model')
        if button0 and self.flag == False:
            self.flag = True
            st.write("Searching best model by {}...".format(deduction_by))
            model, train_score, test_score, pr, re, f1 = pipeline_runner(self.FILE_PATH, deduction_by=deduction_by, target_col_name=self.TARGET,test_size=test_size)
            st.write("Searching done")
            st.write(model)
            st.write("train_score={}, test_score={}, precision={}, recall={}, f1 score={}".format(train_score,test_score,pr,re,f1))
            
            with open("classifier.pkl".format(model,train_score,test_score,pr,re,f1), 'wb') as file:
                pickle.dump(model, file)
            self.download_model(model_bytes=model)

        button1 = col2.button('Random Forest Classifier')
        if button1 and self.flag == False:
            self.flag = True
            model, train_score, test_score, pr, re, f1 = Train(RandomForestClassifier(), X, y).train_model(n_estimators=n_estimators, 
                                                                max_depth=max_depth, min_samples_split=min_samples_split, 
                                                                min_samples_leaf=min_samples_leaf, max_features=max_features,
                                                                bootstrap=bootstrap)
            st.write(model)
            st.write("train_score={}, test_score={}, precision={}, recall={}, f1 score={}".format(train_score,test_score,pr,re,f1))
            
            with open("classifier.pkl".format(model,train_score,test_score,pr,re,f1), 'wb') as file:
                pickle.dump(model, file)
            self.download_model(model_bytes=model)

        button2 = col3.button('Decision Tree Classifier')
        if button2 and self.flag == False:
            self.flag = True
            model, train_score, test_score, pr, re, f1 = Train(DecisionTreeClassifier(), X, y).train_model(max_depth=max_depth, 
                                                                min_samples_split=min_samples_split, 
                                                                min_samples_leaf=min_samples_leaf)
            st.write(model)
            st.write("train_score={}, test_score={}, precision={}, recall={}, f1 score={}".format(train_score,test_score,pr,re,f1))
            with open("classifier.pkl".format(model,train_score,test_score,pr,re,f1), 'wb') as file:
                pickle.dump(model, file)
            self.download_model(model_bytes=model)

        button3 = col4.button('Logistic Regression')
        if button3 and self.flag == False:
            self.flag = True
            model, train_score, test_score, pr, re, f1 = Train(LogisticRegression(), X, y).train_model(C=C_value_log, 
                                                                                                        penalty=penalty_type,
                                                                                                        solver=solver_type)
            st.write(model)
            st.write("train_score={}, test_score={}, precision={}, recall={}, f1 score={}".format(train_score,test_score,pr,re,f1))
            
            with open("classifier.pkl".format(model,train_score,test_score,pr,re,f1), 'wb') as file:
                pickle.dump(model, file)
            self.download_model(model_bytes=model)

        button4 = col5.button('Support Vector Classifier')
        if button4 and self.flag == False:
            self.flag = True
            model, train_score, test_score, pr, re, f1 = Train(SVC(), X, y).train_model(C=C_value_log, 
                                                                                        degree=degree_value, 
                                                                                        kernel=kernel_type)
            st.write(model)
            st.write("train_score={}, test_score={}, precision={}, recall={}, f1 score={}".format(train_score,test_score,pr,re,f1))
            
            with open("classifier.pkl".format(model,train_score,test_score,pr,re,f1), 'wb') as file:
                pickle.dump(model, file)
            self.download_model(model_bytes=model)
