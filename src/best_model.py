import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression

class BestModel:
    """Class for model evaluation and deduction.

    Attributes:
        X_train (np.ndarray): Training data features.
        X_test (np.ndarray): Testing data features.
        y_train (np.ndarray): Training data target values.
        y_test (np.ndarray): Testing data target values.

    Methods:
        __init__: Initializes training and testing data for model evaluation.
        validation: Evaluates the performance of the model using cross-validation, training, and testing data.
        iterator: Iterates over algorithms, evaluates their performance, and stores the results.
        deduction: Selects the best model based on a specified validation method.
    """
    algos = {'rf': RandomForestClassifier(), 'dt': DecisionTreeClassifier(), 'logr': LogisticRegression(), 'svm': SVC()}

    def __init__(self, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> None:
        """Args:
        X_train (np.ndarray): Features of the training data.
        X_test (np.ndarray): Features of the testing data.
        y_train (np.ndarray): Target values of the training data.
        y_test (np.ndarray): Target values of the testing data.
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.X = np.concatenate([X_train, X_test], axis=0)
        self.y = np.concatenate([y_train, y_test], axis=0)

    def validation(self, algo: ClassifierMixin) -> tuple:
        """Evaluates the model performance using cross-validation, training, and testing data.

        Args:
        algo (ClassifierMixin): The algorithm to evaluate.

        Returns:
        tuple: Contains cross-validation score, training score, and testing score.
        """
        logging.debug(f"Validating {algo} algorithm.")
        print("Validation started...")
        cross_val_scores_list = cross_val_score(algo, self.X, self.y)
        cross_val_avg = np.mean(cross_val_scores_list)

        model = algo
        model.fit(self.X_train, self.y_train)
        train_score = model.score(self.X_train, self.y_train)
        test_score = model.score(self.X_test, self.y_test)

        logging.info(f"{algo} - Train Score: {train_score}, Test Score: {test_score}, Cross-Val Score: {cross_val_avg}")
        logging.debug(f"Validation for {algo} completed.")
        return cross_val_avg, train_score, test_score

    def iterator(self) -> dict:
        """Iterates over the algorithms and evaluates their performance.

        Returns:
        dict: A dictionary containing the scores of executed algorithms.
        """
        results = {}
        for key, algo in self.algos.items():
            score = self.validation(algo)
            results[key] = score
        return results

    def deduction(self, deduction_by='test') -> dict:
        """Selects and saves the best model based on a specified validation method.

        Args:
        deduction_by (str): Method of selection ('train', 'test', 'cross').

        Returns:
        dict: Models with scores, sorted based on the chosen method.
        """
        logging.debug(f"Deducting by {deduction_by}...")
        results = self.iterator()
        temp_dict = {}

        for key, val in results.items():
            if deduction_by == 'test':
                temp_dict[key] = val[2]  # Selecting based on test score
            elif deduction_by == 'train':
                temp_dict[key] = val[1]  # Selecting based on train score
            else:
                temp_dict[key] = val[0]  # Selecting based on cross-validation score

            logging.info(f"Searching for the best model based on {deduction_by} score of {key}: {temp_dict[key]}")
            print(f"Searching for the best model based on {deduction_by} score of {key}: {temp_dict[key]}")

        logging.debug("Deduction completed.")
        algos_score = dict(sorted(temp_dict.items(), key=lambda item: item[1], reverse=True))
        return algos_score
