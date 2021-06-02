import pytest
import numpy as np
from sklearn.model_selection import train_test_split

from classification_model.config import core  
from classification_model.processing import utils

@pytest.fixture(scope="session")
def pipeline_inputs():

    # read the data
    data = utils.load_dataset(core.config.app_config.TRAINING_DATA_FILE)

    # Split in X and y
    X = data.drop(labels=core.config.model_config.TARGET_FEATURE_NAME, axis=1)
    y = data[core.config.model_config.TARGET_FEATURE_NAME]

    # For the 2 classes classification
    y = np.where(y=="functional","functional","non functional or functional needs repair")

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=core.config.model_config.SEED,test_size=core.config.model_config.TEST_SIZE)

    return X_train, X_test, y_train, y_test

@pytest.fixture(scope="session")
def pipeline_inputs_tests():

    # read the data
    data = utils.load_dataset(core.config.app_config.TESTING_DATA_FILE)

    # Split in X and y
    X = data.drop(labels=core.config.model_config.TARGET_FEATURE_NAME, axis=1)
    y = data[core.config.model_config.TARGET_FEATURE_NAME]

    # For the 2 classes classification
    y = np.where(y=="functional","functional","non functional or functional needs repair")


    return X,y