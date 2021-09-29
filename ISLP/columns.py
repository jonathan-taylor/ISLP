from typing import NamedTuple, Any
from copy import copy

import numpy as np
from sklearn.base import clone
from sklearn.preprocessing import (OneHotEncoder,
                                   OrdinalEncoder)

class Column(NamedTuple):

    idx: Any
    name: str
    is_categorical: bool = False
    is_ordinal: bool = False
    columns: tuple = ()
    encoder: Any = None
    
    @property
    def column_names(self):
        if not self.is_categorical or self.is_ordinal:
            return [self.name]
        else:
            return ['{0}[{1}]'.format(self.name, c) for c in self.columns]

    def get_columns(self, X):
        
        col = _get_column(self.idx, X, twodim=self.encoder is not None)
        if self.encoder is not None:
            col = self.encoder.transform(col)
        return np.asarray(col)

def _get_column(idx, X, twodim=False):
    if isinstance(X, np.ndarray):
        col = X[:,idx]
    else: # assuming pd.DataFrame
        col = X[idx]
    if twodim and np.asarray(col).ndim == 1:
        return np.asarray(col).reshape((-1,1))
    return col
    
def _get_column_info(X,
                     columns,
                     is_categorical,
                     is_ordinal,
                     default_encoders={'categorical': OneHotEncoder(drop='first', sparse=False),
                                       'ordinal': OrdinalEncoder()}):
    column_info = {}
    for i, col in enumerate(columns):
        name = '{0}'.format(col)
        Xcol = _get_column(col, X, twodim=True)
        if is_categorical[i]:
            if is_ordinal[i]:
                encoder = clone(default_encoders['ordinal'])
                encoder.fit(Xcol)
                columns = ['Ord({0})'.format(col)]
            else:
                encoder = clone(default_encoders['categorical'])
                cols = encoder.fit_transform(Xcol)
                columns = ['Cat({0})[{1}]'.format(col, c) for c in range(cols.shape[1])]

            column_info[col] = Column(col,
                                      name,
                                      is_categorical[i],
                                      is_ordinal[i],
                                      columns,
                                      encoder)
        else:
            column_info[col] = Column(col,
                                      name)
    return column_info

def test_column_info():

    import pandas as pd

    df = pd.DataFrame(np.random.standard_normal((50, 5)), columns=['Aa', 'B', 'Ccc', 'D', 'E'])
    df['E'] = pd.Categorical(np.random.choice(range(4,8), 50, replace=True))

    print(_get_column_info(df,
                           df.columns,
                           [False]*4+[True],
                           [False]*5))

if __name__ == "__main__":
    test_column_info()
