import os
import pandas as pd
import streamlit as st
import numpy as np
import logging

class ShowData:
    """
    Handles data preview, model selection, and predictions.

    Attributes:
        SHOW_DATA_PATH (str): The path to the data file for display.
        model_names (list): List of model names found in the 'models' directory.

    Methods:
        __init__: Initializes the PredictSection instance.
        data_showroom: Displays the first 5 rows of a predefined data file.
        upload_csv: Uploads a CSV file from the user and returns it as a DataFrame.
        prediction: Generates predictions using the provided model and data.
        models_showroom: Presents all saved models for selection and prediction.
        manager: Orchestrates the data display and model selection process.

    """

    SHOW_DATA_PATH = "application_src\\test_data.csv"

    def data_showroom(self) -> None:
        """
        Display the first 5 rows of the predefined data file.

        """
        logging.debug("Data showroom initiated...")
        data = pd.read_csv(self.SHOW_DATA_PATH)
        st.write(data.head())
        logging.info("Data showroom completed.")
