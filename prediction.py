import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from typing import Union
from sklearn.base import ClassifierMixin
import logging

class PredictSection:
    """
    Handles data preview, model selection, and predictions.

    Attributes:
        SHOW_DATA_PATH (str): The path to the data file for display.
        model_names (list): List of model names found in the 'models' directory.

    Methods:
        __init__: Initializes the PredictSection instance.
        data_showrooms: Displays the first 5 rows of a predefined data file.
    """

    SHOW_DATA_PATH = "Data\\test_data.csv"

    def __init__(self, path: str):
        """
        Initializes the PredictSection object.

        Args:
            path (str): The path to the dataset for display.

        """
        self.path = path

    def data_showrooms(self) -> None:
        """
        Display the first 5 rows of the predefined data file.

        """
        st.subheader("Data Preview", divider='rainbow')
        data = pd.read_csv(self.SHOW_DATA_PATH)
        st.write(data.head())
