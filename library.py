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
  
class PearsonTransformer(BaseEstimator, TransformerMixin):
  
  def __init__(self, threshold):
    self.threshold = threshold
  
  def transform(self, X):
    df = X.copy() #your masked table here - see below
    abs_df = df.abs()

    masked_cols = df.columns
    masked_indexes = df.index.values

    masked_df = np.where(abs_df > threshold, True, False)
    masked_df = pd.DataFrame(masked_df, columns = masked_cols, index = masked_indexes)

    upper_mask = np.triu(masked_df, k=1)

    cors = np.where(upper_mask == True)
    cors = np.unique(cors[1])
    correlated_columns = []
    for col in cors:
      correlated_columns.append(masked_cols[col])
    X_ = transformed_df.drop(columns=correlated_columns)

    return X_


  def fit(self, X, y=None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return X

  def fit_transform(self, X, y = None):
    X_ = self.transform(X)
    return X_
  
# Helper method for sigma3 transformer
def compute_3sigma_boundaries(df, column_name):
  assert isinstance(df, pd.core.frame.DataFrame), f'expected Dataframe but got {type(df)} instead.'
  assert column_name in df.columns.to_list(), f'unknown column {column_name}'
  assert all([isinstance(v, (int, float)) for v in df[column_name].to_list()])

  mean = df[column_name].mean()
  sigma = df[column_name].std()
  low_bound = mean - 3*sigma
  high_bound = mean + 3*sigma

  return (low_bound, high_bound)

class Sigma3Transformer(BaseEstimator, TransformerMixin):

  def __init__(self, column):
    self.column = column

  def transform(self, X):
    X_ = X.copy()
    (lowb, highb) = compute_3sigma_boundaries(X_, self.column)
    X_[(self.column)] = X_[self.column].clip(lower = lowb, upper = highb)
    return X_

  def fit(self, X, y=None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return X

  def fit_transform(self, X, y=None):
    X_ = self.transform(X)
    return X_

class TukeyTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, column, action):
    self.column = column
    assert action in ['inner', 'outer'], f'{self.__class__.__name__} action {action} not in ["inner", "outer"]'
    self.action = action

  def transform(self, X):
    X_ = X.copy()
    q1 = X_[self.column].quantile(0.25)
    q3 = X_[self.column].quantile(0.75)
    iqr = q3-q1


    if self.action == 'inner':
      high_fence = q3+1.5*iqr
      low_fence = q1-1.5*iqr
    else:
      high_fence = q3+3*iqr
      low_fence = q1-3*iqr
      
    print(f'{self.action}: high fence: {high_fence}, low fence: {low_fence}\n')
    X_[self.column] = X_[self.column].clip(lower = low_fence, upper=high_fence)

    return X_

  def fit(self, X, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return X

  def fit_transform(self, X, y = None):
    X_ = self.transform(X)
    return X_

class MinMaxTransformer(BaseEstimator, TransformerMixin):
  def __init__(self):
    pass  #takes no arguments

  def transform(self, X):
    X_ = X.copy()
    columns = X_.columns

    for col in columns:
      mi = X_[col].min()
      ma = X_[col].max()
      denom = (ma-mi)
      X_[col] -= mi
      X_[col] /= denom

    return X_

  def fit(self, X, y=None):
    return X

  def fit_transform(self, X, y=None):
    
    X_ = self.transform(X)
    return X_
  
class KNNTransformer(BaseEstimator, TransformerMixin):
  def __init__(self,n_neighbors = 5, weights = "uniform"):
    self.n_neighbors = n_neighbors
    assert weights in ["uniform", "distance"], f'{self.__class__.__name__} action {weights} not in ["uniform", "distance"]'
    self.weights = weights

  def fit(self, X, y=None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return X

  def transform(self, X):
    # X_ = X.copy()
    imputer = KNNImputer(n_neighbors=5,        #a rough guess
                     weights="uniform", add_indicator=False)
    # by default this returns a copy of the input df
    columns = X.columns
    imputed_data = imputer.fit_transform(X)
    X_ = pd.DataFrame(imputed_data, columns = columns)
    return X_

  def fit_transform(self, X, y=None):
    X_ = self.transform(X)
    return X_
