from pathlib import Path

from classification_model.config import core
import pytest
from pydantic import ValidationError

TEST_CONFIG_TEXT = """ 
# Package Overview
PACKAGE_NAME: classification_model

# Data Files
TRAINING_DATA_FILE: train.csv
TESTING_DATA_FILE: test.csv

MODEL_PIPELINE_NAME: gbm_classification_output_v

# Variables
# The variable we are attempting to predict (sale price)
TARGET_FEATURE_NAME: status_group

VARIABLES_TO_KEEP:
  - amount_tsh
  - gps_height
  - construction_year
  - population
  - region
  - basin
  - public_meeting
  - scheme_management
  - permit
  - extraction_type_class
  - management_group
  - payment_type
  - quality_group
  - quantity_group
  - source_type
  - waterpoint_type_group


NUMERICAL_VARIABLES:
  - amount_tsh
  - gps_height
  - construction_year
  - population

YEO_JHONSON_VARIABLES:
  - amount_tsh
  - gps_height
  - population

VARIABLES_TO_GROUP:
  region:
    Dodoma,Singida:
      - Dodoma
      - Singida
    Mara,Tabora,Rukwa,Mtwara,Lindi:
      - MaraTEST
      - Tabora
      - Rukwa
      - Mtwara
      - Lindi
    Manyara,Dar es Salaam,Tanga:
      - Manyara
      - Dar es Salaam
      - Tanga

# Categorical variables (no boolean)
REAL_CATEGORICAL_VARIABLES:
  - region
  - basin
  - scheme_management
  - extraction_type_class
  - management_group
  - payment_type
  - quality_group
  - quantity_group
  - source_type
  - waterpoint_type_group
  - hello_fresh

SEED : 50

ACCEPTABLE_MODEL_DIFFERENCE : 0.05

VARIABLES_THRESHOLD:
  scheme_management: 0.04
  extraction_type_class: 0.06
  management_group: 0.06
  quality_group: 0.08
  source_type: 0.17
  waterpoint_type_group: 0.1
  quantity_group: 0.1

# set train/test split
TEST_SIZE: 0.2

# loss function to be optimized
LOSS_FUNCTION: neg_log_loss

ALLOWED_LOSS_FUNCTIONS:
  - neg_log_loss

"""


INVALID_TEST_CONFIG_TEXT = """ 
# Package Overview
PACKAGE_NAME: classification_model

# Data Files
TRAINING_DATA_FILE: train.csv
TESTING_DATA_FILE: test.csv

MODEL_PIPELINE_NAME: gbm_classification_output_v

# Variables
# The variable we are attempting to predict (sale price)
TARGET_FEATURE_NAME: status_group

VARIABLES_TO_KEEP:
  - amount_tsh
  - gps_height
  - construction_year
  - population
  - region
  - basin
  - public_meeting
  - scheme_management
  - permit
  - extraction_type_class
  - management_group
  - payment_type
  - quality_group
  - quantity_group
  - source_type
  - waterpoint_type_group


NUMERICAL_VARIABLES:
  - amount_tsh
  - gps_height
  - construction_year
  - population

YEO_JHONSON_VARIABLES:
  - amount_tsh
  - gps_height
  - population

VARIABLES_TO_GROUP:
  region:
    Dodoma,Singida:
      - Dodoma
      - Singida
    Mara,Tabora,Rukwa,Mtwara,Lindi:
      - MaraTEST
      - Tabora
      - Rukwa
      - Mtwara
      - Lindi
    Manyara,Dar es Salaam,Tanga:
      - Manyara
      - Dar es Salaam
      - Tanga

# Categorical variables (no boolean)
REAL_CATEGORICAL_VARIABLES:
  - region
  - basin
  - scheme_management
  - extraction_type_class
  - management_group
  - payment_type
  - quality_group
  - quantity_group
  - source_type
  - waterpoint_type_group
  - hello_fresh

SEED : 50

ACCEPTABLE_MODEL_DIFFERENCE : 0.05

VARIABLES_THRESHOLD:
  scheme_management: 0.04
  extraction_type_class: 0.06
  management_group: 0.06
  quality_group: 0.08
  source_type: 0.17
  waterpoint_type_group: 0.1
  quantity_group: 0.1

# set train/test split
TEST_SIZE: 0.2

# loss function to be optimized
LOSS_FUNCTION: fake_loss_function

ALLOWED_LOSS_FUNCTIONS:
  - neg_log_loss

"""


def test_fetch_config_from_yaml(tmpdir):
    """Test if the config object has the 2 attributes config as expected"""

    # Given
    config_dir = Path(tmpdir)
    # Create a config file in a temporary directory
    temp_config_file_path = config_dir / "sample_config.yml"
    temp_config_file_path.write_text(TEST_CONFIG_TEXT)
    parsed_config = core.fetch_config_from_yaml(cfg_path=temp_config_file_path)

    # When
    config = core.create_and_validate_config(parsed_config=parsed_config)

    # Then
    assert config.model_config
    assert config.app_config


def test_validation_error_raise_for_invalid_config(tmpdir):
    """Test if an error is raised for a config ValidationError"""

    # Given
    config_dir = Path(tmpdir)
    # Create a config file in a temporary directory
    temp_config_file_path = config_dir / "sample_config.yml"
    temp_config_file_path.write_text(INVALID_TEST_CONFIG_TEXT)
    parsed_config = core.fetch_config_from_yaml(cfg_path=temp_config_file_path)

    # When
    with pytest.raises(ValidationError) as e:
        core.create_and_validate_config(parsed_config=parsed_config)

    # Then
    assert "not in the allowed set" in str(e.value)


def test_validation_error_raise_for_missing_field(tmpdir):
    """Test if errors are raised when fields are missing"""

    # Given
    config_dir = Path(tmpdir)
    # Create a config file in a temporary directory
    temp_config_file_path = config_dir / "sample_config.yml"
    temp_config_file_path.write_text("""ACCEPTABLE_MODEL_DIFFERENCE : 0.05""")
    parsed_config = core.fetch_config_from_yaml(cfg_path=temp_config_file_path)

    # When
    with pytest.raises(ValidationError) as e:
        core.create_and_validate_config(parsed_config=parsed_config)

    # Then
    assert "field required" in str(e.value)
    assert "ACCEPTABLE_MODEL_DIFFERENCE" not in str(e.value)
