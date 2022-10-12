import pandas as pd
import numpy as np
from joblib.logger import pprint
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

#This class maps values in a column, numeric or categorical.
class MappingTransformer(BaseEstimator, TransformerMixin):
  
  def __init__(self, mapping_column, mapping_dict:dict):
    assert isinstance(mapping_dict, dict), f'{self.__class__.__name__} constructor expected dictionary but got {type(mapping_dict)} instead.'
    self.mapping_dict = mapping_dict
    self.mapping_column = mapping_column  #column to focus on

  def fit(self, X, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return X

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
    assert self.mapping_column in X.columns.to_list(), f'{self.__class__.__name__}.transform unknown column "{self.mapping_column}"'  #column legit?
    
    #now check to see if all keys are contained in column
    column_set = set(X[self.mapping_column])
    keys_not_found = set(self.mapping_dict.keys()) - column_set
    if keys_not_found:
      print(f"\nWarning: {self.__class__.__name__}[{self.mapping_column}] does not contain these keys as values {keys_not_found}\n")

    #now check to see if some keys are absent
    keys_absent = column_set -  set(self.mapping_dict.keys())
    if keys_absent:
      print(f"\nWarning: {self.__class__.__name__}[{self.mapping_column}] does not contain keys for these values {keys_absent}\n")

    X_ = X.copy()
    X_[self.mapping_column].replace(self.mapping_dict, inplace=True)
    return X_

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result
  
#This class will rename one or more columns.
class RenamingTransformer(BaseEstimator, TransformerMixin):
  # def __init__(self, naming_column, naming_dict:dict):
  def __init__(self, naming_dict:dict):
      assert isinstance(naming_dict, dict), f'{self.__class__.__name__} constructor expected dictionary but got {type(naming_dict)} instead.'
      self.mapping_dict = naming_dict
      # self.mapping_column = naming_column  #column to focus on

  def fit(self, X, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return X

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
    # assert self.naming_column in X.columns.to_list(), f'{self.__class__.__name__}.transform unknown column "{self.mapping_column}"'  #column legit?
      
    keys_contatined = False
    #now check to see if all keys are contained in column
    if set(self.naming_dict.keys()).issubset(X.columns):
      keys_contatined = True
    else:
      print(f"\nWarning: {self.__class__.__name__} dataframe does not contain all keys as column names\n")

    X_ = X.copy()
    X_.rename(columns = self.naming_dict, inplace = True)

    return X_

    def fit_transform(self, X, y = None):
      result = self.transform(X)
      return result
    
class OHETransformer(BaseEstimator, TransformerMixin):
  def __init__(self, target_column, dummy_na=False, drop_first=False):  
    self.col = target_column
    self.dummy_na = dummy_na
    self.drop_first = drop_first

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
    X_ = X.copy()

    # assert the column exists in the input dataframe
    assert self.col in X.columns, f'{self.__class__.__name__}. Input Dataframe has no column named {self.col}'
    X_dummies = pd.get_dummies(X_,
                               prefix=self.col,    #your choice
                               prefix_sep='_',     #your choice
                               columns=[self.col],
                               dummy_na=self.dummy_na,    #will try to impute later so leave NaNs in place
                               drop_first=self.drop_first    #really should be True but could screw us up later
                               )
    return X_dummies

  def fit(self, X, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return X

  def fit_transform(self, X, y = None):
      result = self.transform(X)
      return result
    
# this class will drop specified columns. Can also keep specified columns and drop everything else
class DropColumnsTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, column_list, action='drop'):
    assert action in ['keep', 'drop'], f'{self.__class__.__name__} action {action} not in ["keep", "drop"]'
    self.column_list = column_list
    self.action = action

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
    # note: ideally change this to give a message that states the incorrect column
    
    X_ = X.copy()
    if self.action == 'keep':
      X_does_not_contain = set(self.column_list).difference(X.columns)
      assert len(X_does_not_contain) == 0, f'{self.__class__.__name__}: input dataframe does not contain columns {X_does_not_contain}'
      # assert set(self.column_list).issubset(X.columns), f'{self.__class__.__name__}: input dataframe does not contain all columns in col_list input'
      for col in X_.columns:
        if col not in self.column_list:
          del X_[col]
    else:
      # warning here
      X_does_not_contain = set(self.column_list).difference(X.columns)
      if len(X_does_not_contain) > 0:
        print(f"\nWarning: {self.__class__.__name__} does not contain the following columns to drop: {X_does_not_contain}\n")
        # warnings.warn(f'DropColumnsTransformer does not contain the following columns to drop: {X_does_not_contain}')
        self.column_list = set(self.column_list) - X_does_not_contain

      X_.drop(columns = self.column_list, axis=1, inplace=True)
    
    return X_

  def fit(self, X, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return X

  def fit_transform(self, X, y = None):
    X_ = self.transform(X)
    return X_
