import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import KBinsDiscretizer,MinMaxScaler
from feature_engine.transformation import YeoJohnsonTransformer
from feature_engine.encoding import OneHotEncoder
from feature_engine.discretisation import EqualWidthDiscretiser
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
import logging

# Local source tree imports
from classification_model.processing import preprocessors as pp
from classification_model import config

_logger = logging.getLogger(__name__)

pump_pipeline =Pipeline(steps=[("feature_to_keeper",pp.FeatureKeeper(variables_to_keep=config.VARIABLES_TO_KEEP)),
                         ("missing_imputer", pp.MissingImputer(numerical_variables=config.NUMERICAL_VARIABLES)),
                         ("yeoJohnson",YeoJohnsonTransformer(variables=config.YEO_JHONSON_VARIABLES)),
                         ("discretization",EqualWidthDiscretiser(bins=5, variables=config.NUMERICAL_VARIABLES)),
                         ("categorical_grouper",pp.CategoricalGrouping(config_dict=config.VARIABLES_TO_GROUP)),
                         ("rareCategories_grouper",pp.RareCategoriesGrouping(threshold=config.VARIABLES_THRESHOLD)),
                         ("one_hot_encoder",OneHotEncoder(variables=config.REAL_CATEGORICAL_VARIABLES,drop_last=False)),
                         ("scaler",MinMaxScaler()),
                         ("model",GradientBoostingClassifier(random_state=config.SEED))])



