import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from classification_model import config,predict
from classification_model import __version__ as _version


def test_single_make_prediction():
    """ Test make_prediction function for a single prediction """

    # Given
    dataset_file_path = config.DATASET_DIR/config.TESTING_DATA_FILE
    test_data = pd.read_csv(dataset_file_path)

    single_row = test_data.iloc[:1,:]
    # the make_prediction function is expecting a dict
    single_row_dict = dict(single_row)
    
    # When 
    subject = predict.make_prediction(single_row_dict)

    assert subject.get("predictions")[0] in ["functional","functional","non functional or functional needs repair"]
    assert type(subject.get("predictions")) == np.ndarray
    assert subject.get("predictions").shape == (1,)
    assert subject.get("version") == _version



def test_multiple_make_prediction():
    """ Test make_prediction function for multiple prediction """

    # Given
    dataset_file_path = config.DATASET_DIR/config.TESTING_DATA_FILE
    test_data = pd.read_csv(dataset_file_path)

    multiple_row = test_data
    # the make_prediction function is expecting a dict
    multiple_row_dict = dict(multiple_row)
    
    # When 
    subject = predict.make_prediction(multiple_row_dict)

    assert subject.get("predictions")[0] in ["functional","functional","non functional or functional needs repair"]
    assert type(subject.get("predictions")) == np.ndarray
    assert subject.get("predictions").shape == (test_data.shape[0],)
    assert subject.get("version") == _version
    

