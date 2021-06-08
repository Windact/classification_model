import numpy as np
from sklearn.pipeline import Pipeline
from classification_model import pipeline
from classification_model.config import core


def test_pipeline_preprocessors_output(pipeline_inputs):
    """Testing the pipeline output shape before model"""
    print("*******************FUCKER****************************")
    print(f"fucker : {core.config.model_config.VARIABLES_THRESHOLD}")

    # Given
    X_train, X_test, y_train, y_test = pipeline_inputs
    pipeline_before_model = Pipeline(pipeline.pump_pipeline.steps[:-1])

    # When
    subject = pipeline_before_model.fit_transform(X_train, y_train)
    subject_min = np.min(subject, axis=0)
    subject_max = np.max(subject, axis=0)
    # Then
    assert subject.shape[1] == 66
    assert np.sum(subject_min, axis=0) == 0
    assert np.sum(subject_max, axis=0) == subject.shape[1]


def test_pipeline_output(pipeline_inputs):
    """Test the pipeline output"""

    # Given
    X_train, X_test, y_train, y_test = pipeline_inputs
    pipeline.pump_pipeline.fit(X_train, y_train)

    # When
    subject = pipeline.pump_pipeline.predict(X_train)

    # Then
    assert type(subject) == np.ndarray
    assert subject.shape == y_train.shape
    assert set(subject) == set(y_train)
