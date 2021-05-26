import pandas as pd
import sklearn
import joblib

from classification_model import core,pipeline
from classification_model import __version__ as _version
from classification_model.processing import utils


def test_load_dataset():
    """ Testing the loading dataset function """

    # Given
    dataset_file_name = core.config.app_config.TESTING_DATA_FILE

    # When
    subject = utils.load_dataset(filename=dataset_file_name)

    # Then
    assert isinstance(subject,pd.DataFrame)
    assert subject.shape == (5940, 41)

def test_save_pipeline():
    """ Testing the save_pipeline function """

    # Given
    try:
        pipeline_for_test = joblib.load(core.TRAINED_MODEL_DIR/f"{core.config.app_config.MODEL_PIPELINE_NAME}{_version}.pkl")
        subject_file_name = f"{core.config.app_config.MODEL_PIPELINE_NAME}{_version}.pkl"
    except :
        pipeline_ = pipeline.pump_pipeline 
        subject_file_name = f"fake_pipe_line_model_v{_version}.pkl"

    # When
    utils.save_pipeline(pipeline_for_test,subject_file_name)

    # Then
    # Get the files in the model save's directory
    trained_model_dir_file_list = [file.name for file in core.TRAINED_MODEL_DIR.iterdir()]
    
    # Check if the pipeline was saved in TRAINED_MODEL_DIR and with the right filename
    assert subject_file_name in trained_model_dir_file_list
    # Check if the __init__.py file is in the TRAINED_MODEL_DIR
    assert "__init__.py" in trained_model_dir_file_list
    # Check if the TRAINED_MODEL_DIR folder contains just the new saved pipeline and the __init__.py file
    assert len(trained_model_dir_file_list) == 2
    # remove the fake pipeline
    if subject_file_name == f"fake_pipe_line_model_v{_version}.pkl":
        core.TRAINED_MODEL_DIR/subject_file_name.unlink()

def test_load_pipeline():
    """ Testing the load_pipeline function """

    # Given
    pipeline_file_name = f"{core.config.app_config.MODEL_PIPELINE_NAME}{_version}.pkl"

    # When
    subject = utils.load_pipeline(file_name= pipeline_file_name)

    # Then
    assert isinstance(subject,sklearn.pipeline.Pipeline)


def test_remove_old_pipelines():
    """ Test the remove_old_pipelines function """

    former_version_test = "1.0.0"
    latest_version_test = "2.0.0"

    pipeline_for_test = pipeline.pump_pipeline

    # Saving the former pipeline
    save_former_name = f"former_pipeline_v{former_version_test}.pkl"
    save_former_path = core.TRAINED_MODEL_DIR/save_former_name
    joblib.dump(pipeline_for_test,save_former_path)

    trained_model_dir_file_list = [file.name for file in core.TRAINED_MODEL_DIR.iterdir()]

    assert save_former_name in trained_model_dir_file_list

    # Saving the subject pipeline
    subject = f"subject_pipeline_v{latest_version_test}.pkl"
    save_subject_test_path = core.TRAINED_MODEL_DIR/subject
    joblib.dump(pipeline_for_test,save_subject_test_path)

    # When 
    utils.remove_old_pipelines(files_to_keep=[subject,f"{core.config.app_config.MODEL_PIPELINE_NAME}{_version}.pkl"])

    trained_model_dir_file_list = [file.name for file in core.TRAINED_MODEL_DIR.iterdir()]

    # Then
    assert subject in trained_model_dir_file_list
    assert "__init__.py" in trained_model_dir_file_list 
    if f"{core.config.app_config.MODEL_PIPELINE_NAME}{_version}.pkl" in trained_model_dir_file_list:
        assert len(trained_model_dir_file_list) == 3
    else:
        assert len(trained_model_dir_file_list) == 2
    # removing the fake pipeline
    save_subject_test_path.unlink()



def test_input_data_is_valid():
    """ Test input_data_is_valid function """

    # read the test data
    test_data = pd.read_csv(core.DATASET_DIR/core.config.app_config.TESTING_DATA_FILE)

    # Given
    # Droping a feature
    test_data_missing_feature = test_data.copy()
    test_data_missing_feature = test_data.drop(labels=core.config.model_config.VARIABLES_TO_KEEP[0], axis=1)

    # Converting a numerical variable to a string
    test_data_num_type = test_data.copy()
    test_data_num_type[core.config.model_config.NUMERICAL_VARIABLES[0]] = str(test_data_num_type[core.config.model_config.NUMERICAL_VARIABLES[0]])

    #When
    # subject with the expected dataset type and features
    subject = utils.input_data_is_valid(input_data=test_data)
    # subject with missing features
    subject_missing = utils.input_data_is_valid(input_data=test_data_missing_feature)
    # subject with the converted numerical variable
    subject_num_type = utils.input_data_is_valid(input_data=test_data_num_type)


    #Then
    assert subject == True
    assert subject_missing == False
    assert subject_num_type == False

