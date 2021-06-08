import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import MinMaxScaler
from feature_engine.transformation import YeoJohnsonTransformer
from feature_engine.encoding import OneHotEncoder
from feature_engine.discretisation import EqualWidthDiscretiser
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
import logging

# Local source tree imports
from classification_model.processing import preprocessors as pp
from classification_model.config import core

_logger = logging.getLogger(__name__)

pump_pipeline = Pipeline(
    steps=[
        (
            "feature_to_keeper",
            pp.FeatureKeeper(
                variables_to_keep=core.config.model_config.VARIABLES_TO_KEEP
            ),
        ),
        (
            "missing_imputer",
            pp.MissingImputer(
                numerical_variables=core.config.model_config.NUMERICAL_VARIABLES
            ),
        ),
        (
            "yeoJohnson",
            YeoJohnsonTransformer(
                variables=core.config.model_config.YEO_JHONSON_VARIABLES
            ),
        ),
        (
            "discretization",
            EqualWidthDiscretiser(
                bins=5, variables=core.config.model_config.NUMERICAL_VARIABLES
            ),
        ),
        (
            "categorical_grouper",
            pp.CategoricalGrouping(
                config_dict=core.config.model_config.VARIABLES_TO_GROUP
            ),
        ),
        (
            "rareCategories_grouper",
            pp.RareCategoriesGrouping(
                threshold=core.config.model_config.VARIABLES_THRESHOLD
            ),
        ),
        (
            "one_hot_encoder",
            OneHotEncoder(
                variables=core.config.model_config.REAL_CATEGORICAL_VARIABLES,
                drop_last=False,
            ),
        ),
        ("scaler", MinMaxScaler()),
        (
            "model",
            GradientBoostingClassifier(random_state=core.config.model_config.SEED),
        ),
    ]
)
