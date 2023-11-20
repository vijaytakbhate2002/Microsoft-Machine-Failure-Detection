import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import logging
import os

class DataProcessing:
    """Class for data preprocessing.

    Methods:
        __init__: Constructor to set the path and initial data.
        ingest: Reads data from a specified path and returns a DataFrame.
        preprocess_data: Processes data by performing MinMax scaling.

    Attributes:
        path (str): The path to the data file.
        data (pd.DataFrame): The data for processing (if provided).
    """

    def __init__(self, path: str = None, data: pd.DataFrame = None) -> None:
        """Constructor to set the path and initial data.

        Args:
            path (str): The path to the data file.
            data (pd.DataFrame): The data for processing (if provided).
        """
        self.path = path
        self.data = data

    def ingest(self) -> pd.DataFrame:
        """Reads data from the specified path and returns a DataFrame.

        Returns:
            pd.DataFrame: The ingested data.
        """
        logging.debug("Data ingestion started...")
        try:
            data = pd.read_csv(self.path)
        except FileNotFoundError:
            print(f"{self.path} does not exist")
            logging.error(f"{self.path} does not exist")
            return pd.DataFrame()  # Return empty DataFrame if file not found

        logging.info("Data ingestion completed.")
        return data

    def preprocess_data(self) -> pd.DataFrame:
        """Processes the data using MinMax scaling.

        Returns:
            pd.DataFrame: The processed data after MinMax scaling.
        """
        if self.data is not None:
            data = self.data
        else:
            if os.path.exists(self.path):
                data = self.ingest()
            else:
                print(f"Specified path {self.path} does not exist")
                logging.error(f"Specified path {self.path} does not exist")
                return pd.DataFrame()  # Return empty DataFrame if path doesn't exist

        columns = data.columns
        logging.debug("MinMax scaling started...")
        scaler = MinMaxScaler()
        data = scaler.fit_transform(data)
        logging.debug("MinMax scaling completed.")
        
        data = pd.DataFrame(data, columns=columns)
        logging.info("Data processing completed.")
        return data
