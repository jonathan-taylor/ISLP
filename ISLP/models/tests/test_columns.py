import numpy as np, pandas as pd
from sklearn.base import clone

from pandas.api.types import CategoricalDtype
from ..columns import _get_column_info

def test_column_info():

    cat_type = CategoricalDtype(categories=list("abcd"), ordered=True)
    df = pd.DataFrame(np.random.standard_normal((50, 5)), columns=['Aa', 'B', 'Ccc', 'D', 'E'])
    df['E'] = pd.Categorical(np.random.choice(range(4,8), 50, replace=True))
    df['D'] = pd.Categorical(np.random.choice(['a','b','c','d'], 50, replace=True))
    df['D'].astype(cat_type)
    print(_get_column_info(df,
                           df.columns,
                           [False]*4+[True],
                           [False]*5))

