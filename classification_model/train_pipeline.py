import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
import joblib
import logging

from classification_model.config import core
from classification_model import pipeline
from classification_model import __version__ as _version 
from classification_model.processing import utils

_logger = logging.getLogger(__name__)


def run_training():
    """ Train the model """

    # read the data
    data = utils.load_dataset(core.config.app_config.TRAINING_DATA_FILE)

    # Split in X and y
    X = data.drop(labels=core.config.model_config.TARGET_FEATURE_NAME, axis=1)
    y = data[core.config.model_config.TARGET_FEATURE_NAME]

    # For the 2 classes classification
    y = np.where(y=="functional","functional","non functional or functional needs repair")

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=core.config.model_config.SEED,test_size=core.config.model_config.TEST_SIZE)


    # Training wtih gridsearch
    parameters = {'model__learning_rate':[0.1,0.3,0.5],'model__n_estimators': [100,120,150],'model__subsample': [0.6,0.8,1],'model__max_depth': [6,8,10],'model__min_samples_split':[4,6,10]}
    clf = GridSearchCV(pipeline.pump_pipeline, parameters, scoring=core.config.model_config.LOSS_FUNCTION,n_jobs=-1,cv=3,refit=True,verbose=2)
    clf.fit(X_train, y_train)
    _logger.info(f"Best parameters : {clf.best_params_}")

    #utils.show_results(clf,X_train_transformed,X_test_transformed,y_train,y_test)

    # Report
    utils.show_results(clf.best_estimator_,X_train,X_test,y_train,y_test)
    
    # Saving the model pipeline
    utils.save_pipeline(clf.best_estimator_)

if __name__ == '__main__':
    run_training()


