import pandas as pd
import pathlib
import typing as t
from pydantic import BaseModel, validator
from strictyaml import load, YAML

import classification_model



pd.options.display.max_rows = 10
pd.options.display.max_columns = 10


PACKAGE_ROOT = pathlib.Path(classification_model.__file__).resolve().parent
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"
DATASET_DIR = PACKAGE_ROOT / "datasets"


class AppConfig(BaseModel):
    """ Application-level config """

    PACKAGE_NAME: str
    MODEL_PIPELINE_NAME: str
    TRAINING_DATA_FILE: str
    TESTING_DATA_FILE: str

class ModelConfig(BaseModel):
    """
    All configuration relevant to model
    training and feature engineering.
    """

    TARGET_FEATURE_NAME: str
    VARIABLES_TO_KEEP: t.Sequence[str]
    NUMERICAL_VARIABLES: t.Sequence[str]
    YEO_JHONSON_VARIABLES: t.Sequence[str]
    VARIABLES_TO_GROUP: t.Dict
    REAL_CATEGORICAL_VARIABLES: t.Sequence[str]
    SEED: int 
    ACCEPTABLE_MODEL_DIFFERENCE: float
    VARIABLES_THRESHOLD: t.Dict[str,float]
    TEST_SIZE: float
    # the order is necessary for validation
    ALLOWED_LOSS_FUNCTIONS: t.Sequence[str]
    LOSS_FUNCTION: str

    @validator("LOSS_FUNCTION")
    def allowed_loss_function(cls, value, values):
        """
        Loss function to be optimized.
        neg_log_loss refers to a negative log loss.
        """

        allowed_loss_functions = values.get("ALLOWED_LOSS_FUNCTIONS")
        if value in allowed_loss_functions:
            return value
        raise ValueError(
            f"the loss parameter specified: {value}, "
            f"is not in the allowed set: {allowed_loss_functions}"
        )
        
    @validator("VARIABLES_THRESHOLD")
    def variable_threshold_range(cls,value):
        print(value)
        check_list = []
        for k,v in value.items():
            if not (v >= 0 and v <=1):
                check_list.append((k,v))
        if len(check_list) == 0:
            return value
        else:
            raise ValueError(f"The following key value pairs do not respect the range of the thresholds : {check_list}")
                
class Config(BaseModel):
    """Master config object."""

    app_config: AppConfig
    model_config: ModelConfig

def find_config_file() -> pathlib.Path:
    """Locate the configuration file."""
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise Exception(f"Config not found at {CONFIG_FILE_PATH!r}")

def fetch_config_from_yaml(cfg_path: pathlib.Path = None) -> YAML:
    """Parse YAML containing the package configuration."""

    if not cfg_path:
        cfg_path = find_config_file()

    if cfg_path:
        with open(cfg_path, "r") as conf_file:
            parsed_config = load(conf_file.read())
            return parsed_config
    raise OSError(f"Did not find config file at path: {cfg_path}")

def create_and_validate_config(parsed_config: YAML = None) -> Config:
    """Run validation on config values."""
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()

    # specify the data attribute from the strictyaml YAML type.
    _config = Config(
        app_config=AppConfig(**parsed_config.data),
        model_config=ModelConfig(**parsed_config.data),
    )

    return _config

config = create_and_validate_config()
