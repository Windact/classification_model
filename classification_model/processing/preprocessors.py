# System imports
import pandas as pd
import numpy as np
import os
# Third-party imports
from sklearn.base import BaseEstimator,TransformerMixin


# Drop useless columns
class FeatureKeeper(BaseEstimator, TransformerMixin):
    """  
    Extract the features that are given in the variables_to_keep parameters from a dataset

    Attributes
    ----------
    variables_to_keep : list, optional
        a list of variables to extract.

    Methods
    -------
    fit(X,y=None)
        Returns self
    transform(X)
        Extract the features inf variables_to_keep from the pd.DataFrame X
    """
    def __init__(self,variables_to_keep=None):
        """ 
        Parameters
        ----------
        variables_to_keep : list, optional
            A list of variables to extract.
        """

        self.variables_to_keep = variables_to_keep
        
    def fit(self,X,y=None):
        """ Returns self

        Parameters
        ----------
        X : pd.DataFrame
            A pd.DataFrame that must contain at least the variables in variables_to_keep
        y : pd.Series, optional

        Return
        ------
        self
        """

        return self
    
    def transform(self,X):
        """ Transform the pd.DataFrame X bu extracting the features in self.variables_to_keep

        Parameters
        ----------
        X : pd.DataFrame
            A pd.DataFrame that must contain at least the variables in variables_to_keep
        
        Return
        ------
        A pd.DataFrame containing only the features in to_variables_to_keep
        """

        X = X.copy()
        X = X[self.variables_to_keep]
        return X
    
# Categorical grouping
class CategoricalGrouping(BaseEstimator,TransformerMixin):
    
    def __init__(self,config_dict = {}):
        
        self.config_dict = config_dict
        
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        X = X.copy()
        for k in self.config_dict.keys():
            for list_key,list_value in self.config_dict[k].items():
                X[k] = np.where(X[k].isin(list_value),list_key,X[k])
        
        return X
    
# Rare categories
class RareCategoriesGrouping(BaseEstimator, TransformerMixin):
    """ Group rare categories base on the treshold given for certain categorical variables 
    
    Attributes
    ----------
    threshold : dict
        A dictionary with the keys being a categorical variable name and the value a float between 0 and 1 represent the threshold. All classes of the variable below that
        value will be grouped under the Rare class 

    cat_dict : dict 
        A dictionary with keys being the categorical variable name and the value a list of classes of that variable that have a frequency below the treshold given in the threshold dictionary

    Methods
    -------
    fit(X,y=None)
        Gets the categories for each variable that are under the frequency threshold.
        Returns self
    transform(X)
        Groups rare categories base on the treshold given for certain categorical variables 
        Returns X 
    """
    
    def __init__(self,threshold= {}):
        """ 
        Parameters
        ----------
        threshold : dict 
            A dictionary with the keys being a categorical variable name and the value a float between 0 and 1 represent the threshold. All classes of the variable below that
            value will be grouped under the Rare class 
        cat_dict : dict 
            A dictionary with keys being the categorical variable name and the value a list of classes of that variable that have a frequency below the treshold given in the threshold dictionary
        """

        self.threshold = threshold
        self.cat_dict = {}
        
    def fit(self, X,y=None):
        """ Returns self

        Parameters
        ----------
        X : pd.DataFrame
            A pd.DataFrame that must contain at least the variables in threshold keys.
        y : pd.Series, optional

        Gets the categories for each variable that are under the frequency threshold.

        Return
        ------
        self
        """

        X = X.copy()
        for k,v in self.threshold.items():
            cat_list = X[k].value_counts(normalize=True)
            self.cat_dict[k] = list(cat_list[cat_list<float(v)].index)
        return self
    
    def transform(self,X):
        """ Groups rare categories base on the treshold given for certain categorical variables 

        Parameters
        ----------
        X : pd.DataFrame
            A pd.DataFrame that must contain at least the variables in threshold keys.
        
        Return
        ------
        A pd.DataFrame with the variables with categories below the threshold grouped under the Rare category
        """
        X = X.copy()
        for k,v in self.cat_dict.items():
            X[k] = np.where(X[k].isin(v),"Rare",X[k])
        
        return X
    
    
    
# Missing values
class MissingImputer(BaseEstimator,TransformerMixin):
    
    def __init__(self,numerical_variables):
        self.numerical_variables = numerical_variables
        
    def fit(self,X,y=None):
        self.imputer_dict = {}
        for v in X.columns:
            if v in self.numerical_variables:
                self.imputer_dict[v] = X[v].mean()
            else:
                self.imputer_dict[v] = X[v].mode()[0]

        return self
    
    def transform(self,X):
        X = X.copy()
        for v in X.columns:
            X[v] = X[v].fillna(self.imputer_dict[v])

        return X
    
    
