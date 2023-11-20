import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
import logging
import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

class Train:
    """Class to handle training machine learning models.

    Methods:
        __init__: Constructor to initialize the best algorithm, features (X), and target variable (y).
        validate: Validates the model with train and test values.
        train_model: Trains the model and returns evaluation metrics.

    Attributes:
        best_algo: The best performing algorithm for training.
        X: Features used for training.
        y: Target variable for prediction.
    """
    def __init__(self, best_algo, X, y) -> None:
        """Constructor to initialize the best algorithm, features, and the target variable.

        Args:
            best_algo (ClassifierMixin): The best performing algorithm for training.
            X (np.ndarray): Features used for training.
            y (np.ndarray): Target variable for prediction.
        """
        self.best_algo = best_algo 
        self.X = X
        self.y = y

    def validate(self, **kwargs) -> tuple:
        """Validates the model with train and test values.

        Args:
            **kwargs: Additional keyword arguments for model setup.

        Returns:
            tuple: Contains training score, testing score, precision, recall, and F1 score.
        """
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2)
        model = self.best_algo.set_params(**kwargs)
        model = model.fit(X_train, y_train)
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)

        logging.info(f"Training score: {train_score}, Testing score: {test_score}")

        y_pred = model.predict(X_test)
        pr = precision_score(y_true=y_test, y_pred=y_pred)
        re = recall_score(y_true=y_test, y_pred=y_pred)
        f1 = f1_score(y_true=y_test, y_pred=y_pred)

        logging.info(f"Precision: {pr}, Recall: {re}, F1 Score: {f1}")

        return train_score, test_score, pr, re, f1

    def train_model(self, **kwargs) -> ClassifierMixin:
        """Trains the model and returns evaluation metrics.

        Args:
            **kwargs: Additional keyword arguments for model setup.

        Returns:
            ClassifierMixin: The trained model.
        """
        train_score, test_score, pr, re, f1 = self.validate(**kwargs)
        model = self.best_algo.set_params(**kwargs)
        model.fit(self.X, self.y)
        return model, train_score, test_score, pr, re, f1
