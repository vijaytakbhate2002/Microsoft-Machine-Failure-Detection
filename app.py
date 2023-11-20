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
import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression
from application_src import visualisation
import prediction
from train_support import ModelTrainer
import contact

log_file = 'model_log.log'
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_log_file()-> None:
    """ clean log file when it reaches to 1MB size 
    """
    if os.path.exists(log_file):
        file_size = os.path.getsize(log_file)
        if file_size > 1000000:
            open("my_log.log", 'w').close()
clean_log_file()

st.title(':blue[Microsoft Machine Failure Detection]')

st.sidebar.title('Navigation')
playground_btn = st.sidebar.button("Playground")
visualisation_btn = st.sidebar.button("Visualisation")
contact_btn = st.sidebar.button("Developer Contact")

if visualisation_btn:
    logging.info("Visualisation button clicked, redirecting to visualization module.")
    visualisation.visual()
    logging.info("Returned from visualisation module.")

elif contact_btn:
    contact.show_contact()

else:
    logging.info("Playground button clicked, initiating model training.")
    ModelTrainer().train()
    logging.info("Model training completed.")
