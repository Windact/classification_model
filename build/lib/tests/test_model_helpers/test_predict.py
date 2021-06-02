import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score,f1_score

from dl_classification_model.predict import make_prediction as nn_make_prediction

from classification_model.config import core
from classification_model import predict
from classification_model import __version__ as _version



def test_single_make_prediction():
    """ Test make_prediction function for a single prediction """

    # Given
    dataset_file_path = core.DATASET_DIR/core.config.app_config.TESTING_DATA_FILE
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
    dataset_file_path = core.DATASET_DIR/core.config.app_config.TESTING_DATA_FILE
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
    


def test_prediction_quality_against_benchmark(pipeline_inputs_tests):
    """ Checking our new model against a benchmark """

    # Given
    X,y = pipeline_inputs_tests
    benchmark_value = pd.Series(y).value_counts(normalize =True)[0]
    y_pred = predict.make_prediction(X)["predictions"]

    # When
    subject = accuracy_score(y,y_pred,normalize=True)

    # Then
    assert subject>benchmark_value



def test_prediction_quality_against_another_model(pipeline_inputs_tests):
    """ Checking our new model is better than another one with the f1_score
    The F1 score can be interpreted as a weighted average of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0.
     """

    # Given
    X,y = pipeline_inputs_tests
    prev_pred = nn_make_prediction(X)["predictions"]
    current_pred = predict.make_prediction(X)["predictions"]

    # Output encoding for the use of the f1_score
    y_encoded = np.where(y=="functional",1,0)
    prev_encoded = np.where(prev_pred=="functional",1,0)
    current_encoded = np.where(current_pred=="functional",1,0)

    prev_f1_score = f1_score(y_encoded,prev_encoded,average="binary")




    # When
    subject_f1_score = f1_score(y_encoded,current_encoded,average="binary")
    # Then
    print("*********************************")
    print(prev_f1_score)
    print(subject_f1_score)
    assert subject_f1_score>prev_f1_score

