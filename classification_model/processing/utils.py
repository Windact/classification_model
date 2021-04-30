import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import logging

from classification_model.config import config
from classification_model import __version__ as _version

import logging

_logger = logging.getLogger(__name__)

def load_dataset(filename):
    """ Load a dataset from the config.DATASET_DIR directory.

    Parameters
    ----------
    filename : str
        A csv file filename
    
    Returns
    -------
    pandas.DataFrame
    """

    _data = pd.read_csv(f"{config.DATASET_DIR/filename}",sep=",", encoding="utf-8")
    _logger.info(f"{filename} was loaded from {config.DATASET_DIR}")
    return _data


def save_pipeline(pipeline_to_save,save_file_name=f"{config.MODEL_PIPELINE_NAME}{_version}.pkl"):
    """ Save a pipeline 
    Save a pipeline and overwrites a pipeline of the same version as this one.

    Parameters
    ----------
    pipeline_to_save : pipeline
        A pipeline with its preprocessing steps and the estimator 
    """
    
    # setting up the versioned filename
    #save_file_name = f"{config.MODEL_PIPELINE_NAME}{_version}.pkl"
    save_file_path = config.TRAINED_MODEL_DIR/save_file_name

    # remove older pipelines
    remove_old_pipelines(files_to_keep=[save_file_name])
    # saving the pipeline
    joblib.dump(pipeline_to_save,save_file_path)
    # logging
    _logger.info(f"The pipeline was saved as : {save_file_name}")

def load_pipeline(file_name):
    """ load a saved pipeline 
    
    Parameters
    ----------
    file_name: str
        The pipeline filename.

    Returns
    -------
    sklearn.pipeline
        A fitted pipeline.
    """

    file_path = config.TRAINED_MODEL_DIR/file_name
    _fitted_pipeline = joblib.load(file_path)
    _logger.info(f"The pipeline {file_name} has been loaded")
    return _fitted_pipeline


def remove_old_pipelines(files_to_keep):
    """  Remove previous pipeline based on the version.

    Do not remove the __init__.py file and log the removed files if there was.
    
    Parameters
    ----------
    files_to_keep : list
        A list of files to not remove.
    """

    # adding __init__.py to the not delete list as we do not want to delete it.
    to_not_remove = files_to_keep + ["__init__.py"]
    removed_files = []
    for file in config.TRAINED_MODEL_DIR.iterdir():
        if file.name not in to_not_remove:
            file.unlink()
            removed_files.append(file.name)
    
    # Logging the removal of files if there was.
    if len(removed_files)>0:
        _logger.info(f"file/pipelines removed : {removed_files}")



# Show the classification report for GBM model or model with classes_ attributes for classes name for the train and test sets.
def show_results(clf,X_train,X_test,y_train,y_test):
    """  print the classification report of a model on the training set and test set.

    Parameters
    ----------
    clf : a classifier estimator, can be a pipeline with an estimator or an estimator. 
        Must have the classes_ attribute for the classes name. 
    X_train : pandas.Dataframe or numpy.array
            The train dataset
    X_test :  pandas.Dataframe or numpy.array
            The test dataset
    y_train : pandas.Series, numpy.array
            Contains the target classes for the training set.
    y_test : pandas.Series, numpy.array
            Contains the target classes for the test set.
    """

    print(f"***** Report for the model *****")
    print(' ***** Train ******')
    y_pred = clf.predict(X_train)
    #print(f"  Accuracy Score : {accuracy_score(y_train,y_pred)}")
    _logger.info(f"Train accuracy score: {accuracy_score(y_train,y_pred)}")
    print(classification_report(y_train, y_pred, target_names=clf.classes_))
    print(' ***** Test ******')
    y_pred = clf.predict(X_test)
    #print(f"  Accuracy Score : {accuracy_score(y_test,y_pred)}")
    _logger.info(f"Test accuracy score: {accuracy_score(y_test,y_pred)}")
    print(classification_report(y_test, y_pred, target_names=clf.classes_))



def input_data_is_valid(input_data):
    """ Check the input data is valid or not. 
    
    Parameters
    ----------
    input_data : pd.DataFrame

    Returns
    -------
    boolean
        A boolean indicating whether the input data is valid (True) or  not (False)
    """

    data = input_data.copy()

    # Check if the features expected are in the input_data
    features_missing = [feature for feature in config.VARIABLES_TO_KEEP if feature not in data.columns]

    if len(features_missing)> 0:
        _logger.error(f"Those features are missing in the input data : {features_missing}")
        return False

    # Check if the features are in their expected type
    input_data_num_var = [var for var in config.NUMERICAL_VARIABLES if data[var].dtype != "O"]

    if len(input_data_num_var) != len(config.NUMERICAL_VARIABLES):
        _logger.error(f"Some of the expected numerical variables are not numerical : {[(var,data[var].dtype) for var in config.NUMERICAL_VARIABLES if var not in input_data_num_var]}")
        return False

    _logger.info("The input data has been validated")
    return True
