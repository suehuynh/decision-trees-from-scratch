import csv
import random
import numpy as np
import pandas as pd

def get_data(filename: str, class_name: str) -> np.ndarray:
    """
    Converts given file contents to binary features.

    Parameters
    ----------
    filename : str
        Float representing probability value.
    class_name : str
        String representing the class name to be converted to binary entries.

    Returns
    -------
    np.ndarray
        Numpy array with size n*(m_1 + m_2 + ... + m_m + 1), the first column has 1 or 0 corresponding to label.
    """
    data = read_data(filename)
    data = convert_to_binary_features(data, class_name)
    return np.array(split_data(data), dtype=object)


def read_data(filename: str) -> list:
    """
    Reads in given file and turns it into a numpy array.

    Parameters
    ----------
    filename : str
        Float representing probability value.

    Returns
    -------
    list
        List containing rows of the data, read from filename.
    """
    data = []
    with open(filename) as f:
        reader = csv.reader(f)
        for row in reader:
            data.append(row)
    return data


def convert_to_binary_features(data: list, class_name: str) -> list:
    """
    Converts the given data into binary entries for the specified class name.

    Parameters
    ----------
    data : list
        List containing rows of the data, read from filename.
    filename : str
        Float representing probability value.

    Returns
    -------
    list
        The data with binary values for the specified column, as a list.
    """
    features = []
    for feature_index in range(0, len(data[0]) - 1):
        feature_values = list(set([obs[feature_index] for obs in data]))
        feature_values.sort()
        if len(feature_values) > 2:
            features.append(feature_values[:-1])
        else:
            features.append([feature_values[0]])
    new_data = []
    for obs in data:
        new_obs = [
            1 if obs[-1] == class_name else 0
        ]  # label = 1 if label in the dataset is won
        for feature_index in range(0, len(data[0]) - 1):
            current_feature_value = obs[feature_index]
            for possible_feature_value in features[feature_index]:
                new_obs.append(current_feature_value == possible_feature_value)
        new_data.append(new_obs)

    return new_data


def split_data(
    data: list, num_training: int = 1000, num_validation: int = 1000
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Performs train-eval-test split on given data.

    Parameters
    ----------
    data : list
        List containing rows of the data.
    num_training : int
        Integer representing the training dataset size, defaulted at 1000.
    num_validation : int
        Integer representing the evaluation dataset size, defaulted at 1000.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Tuple containing the training data, validation data, and test data.
    """
    random.shuffle(data)
    # casting to a numpy array
    data = np.array(data)
    return (
        data[0:num_training],
        data[num_training : num_training + num_validation],
        data[num_training + num_validation : len(data)],
    )

def prepare_data(X: pd.DataFrame, y: pd.Series) -> np.ndarray:
    """
    Standardizes the dataframe for the DecisionTree engine.
    1. Ensures all features are numerical.
    2. Places the target variable (label) in the first column (index 0).
    3. Returns a clean NumPy array.
    """
    target = y.values.reshape(-1, 1).astype(float)
    
    try:
        features = X.values.astype(float)
    except ValueError as e:
        raise ValueError("Ensure all categorical columns are encoded before calling prepare_data.") from e
    
    return np.hstack((target, features))