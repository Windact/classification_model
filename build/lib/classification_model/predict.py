import pandas as pd
import joblib
import logging

from classification_model import __version__ as _version
from classification_model.config import core
from classification_model.processing import utils

_logger = logging.getLogger(__name__)


def make_prediction(input_data):
    """ Predict the class of the water pumps 
    
    Parameters
    ----------
    input_data : dict
         A dictionary containing features (keys) expected for by the pipeline and its corresponding values
    
    Returns
    -------
    dict
    Returns a dictionary containing the predictions and the version of the pipeline
    """

    # Converting the input_data dict to a pd.DataFrame
    data = pd.DataFrame(input_data)
    # Checking if the data is valid
    if utils.input_data_is_valid(data):
        # loading the latest fitted pipeline
        pipe_line_file_name = f"{core.config.app_config.MODEL_PIPELINE_NAME}{_version}.pkl"
        _pipe_pump = utils.load_pipeline(file_name=pipe_line_file_name)
        # Predictions
        outputs = _pipe_pump.predict(data)

        results = {"predictions" : outputs, "version" : _version}

        _logger.info(
            f"Making predictions with the model version : {_version}"
            f"Inputs : {data}"
            f"predictions : {results}"
        )
        return results
        

# if __name__ == '__main__':

#     # Test the pipeline
#     import numpy as np
#     from sklearn.model_selection import train_test_split
#     from sklearn.metrics import accuracy_score

#     data = pd.read_csv(core.TRAIN_DATA_FILE,sep=",", encoding="utf-8")

#     # Split in X and y
#     X = data.drop(labels=core.TARGET_FEATURE_NAME, axis=1)
#     y = data[core.TARGET_FEATURE_NAME]

#     # For the 2 classes classification
#     y = np.where(y=="functional","functional","non functional or functional needs repair")

#     # Train test split
#     X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=core.config.app_config.SEED)

#     # Test
#     pred = make_prediction(X_test)
#     # Scores
#     print(f"Accuracy Score on Test set : {accuracy_score(y_test,pred)}")