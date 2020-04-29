from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import preprocessors as pp
import config


titanic_pipe = Pipeline(
    [
     ('categorical_imputer',
      pp.CategoricalImputer(variables=config.CATEGORICAL_VARS)),
     
     ('missin_indicator',
      pp.MissingIndicator(variables=config.NUMERICAL_VARS)),
     
     ('numerical_imputer',
      pp.NumericalImputer(variables=config.NUMERICAL_VARS)),
     
     ('cabin_extractor',
      pp.ExtractFirstLetter(variables=config.CABIN)),
     
     ('rare_label_encoder',
      pp.RareLabelCategoricalEncoder(variables=config.CATEGORICAL_VARS)),
     
     ('categorical_encoder',
      pp.CategoricalEncoder(variables=config.CATEGORICAL_VARS)),
     
     ('scaler', StandardScaler()),
     
     ('linear_model', LogisticRegression(C=0.0005, random_state=0))
    ]
    )