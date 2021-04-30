import pandas as pd
import pathlib

import classification_model



pd.options.display.max_rows = 10
pd.options.display.max_columns = 10


PACKAGE_ROOT = pathlib.Path(classification_model.__file__).resolve().parent
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"
DATASET_DIR = PACKAGE_ROOT / "datasets"

# data
TRAINING_DATA_FILE = "train.csv"
TESTING_DATA_FILE = "test.csv"
TARGET_FEATURE_NAME = "status_group"


MODEL_PIPELINE_NAME = "gbm_classification_output_v"

# Seed for random state
SEED = 42


VARIABLES_THRESHOLD = {
    'scheme_management': 0.04,
    'extraction_type_class': 0.06,
    'management_group':0.06,
    'quality_group':0.08,
    'source_type':0.17,
    'waterpoint_type_group':0.1,
    'quantity_group':0.1
}

VARIABLES_TO_KEEP = ['amount_tsh', 'gps_height','construction_year',
       'population', 'region', 'basin',
       'public_meeting', 'scheme_management', 'permit',
       'extraction_type_class', 'management_group', 'payment_type',
       'quality_group', 'quantity_group', 'source_type',
       'waterpoint_type_group']

NUMERICAL_VARIABLES = ['amount_tsh', 'gps_height','construction_year','population']
YEO_JHONSON_VARIABLES = ['amount_tsh', 'gps_height','population']

VARIABLES_TO_GROUP = {
    'region':{
        'Dodoma,Singida':["Dodoma","Singida"],
        'Mara,Tabora,Rukwa,Mtwara,Lindi': ["Mara","Tabora","Rukwa","Mtwara","Lindi"],
        'Manyara,Dar es Salaam,Tanga' : ["Manyara","Dar es Salaam","Tanga"]
    }
}

# Gradient boosting machine best params:

GBM_BEST_PARAMS = {
    "max_depth" : 8,
    "min_samples_split" : 6,
    "n_estimators" : 150,
    "subsample" : 0.8
}

# Categorical variables (no boolean)
REAL_CATEGORICAL_VARIABLES = ['region',
 'basin',
 'scheme_management',
 'extraction_type_class',
 'management_group',
 'payment_type',
 'quality_group',
 'quantity_group',
 'source_type',
 'waterpoint_type_group']

