from itertools import product

import pytest

import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression

from ISLP.models import ModelMatrix as MM
from ISLP.models.strategy import min_max, step, validator_from_constraints
from ISLP.models.generic_selector import FeatureSelector

def test_min_max():

    n, p = 100, 7
    X = np.random.standard_normal((n, p))
    Y = np.random.standard_normal(n)
    D = pd.DataFrame(X, columns=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'][:p])
    D['A'] = pd.Categorical(np.random.choice(range(5), (n,), replace=True))

    model_matrix = MM(list(D.columns))
    model_matrix.fit(D)

    strategy = min_max(model_matrix,
                       min_terms=1,
                       max_terms=len(model_matrix.terms),
                       lower_terms=['B','C'],
                       upper_terms=['B','C', 'D', 'H', 'I'])

    min_max_selector = FeatureSelector(LinearRegression(),
                                       strategy,
                                       cv=3)

    min_max_selector.fit(D, Y)

    print(min_max_selector.results_)
    print('selected')
    print(min_max_selector.selected_state_)
    print(min_max_selector.results_[min_max_selector.selected_state_])

def test_step():

    n, p = 100, 7
    X = np.random.standard_normal((n, p))
    Y = np.random.standard_normal(n)
    D = pd.DataFrame(X, columns=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'][:p])
    D['A'] = pd.Categorical(np.random.choice(range(5), (n,), replace=True))

    model_matrix = MM(list(D.columns))
    model_matrix.fit(D)

    for (direction,
         upper_terms,
         lower_terms) in product(['forward',
                                  'backward',
                                  'both'],
                                 [None, ['B','C', 'D', 'H', 'I']],
                                 [None, ['B', 'C']]):

        strategy = step(model_matrix,
                        direction=direction,
                        min_terms=1,
                        max_terms=len(model_matrix.terms),
                        lower_terms=['B','C'])

        step_selector = FeatureSelector(LinearRegression(),
                                        strategy,
                                        cv=3)

        step_selector.fit(D, Y)

        print(step_selector.results_)
        print(step_selector.selected_state_)
    
def test_constraint():

    n, p = 100, 7
    X = np.random.standard_normal((n, p))
    Y = np.random.standard_normal(n)
    D = pd.DataFrame(X, columns=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'][:p])
    D['A'] = pd.Categorical(np.random.choice(range(5), (n,), replace=True))

    model_matrix = MM(list(D.columns))
    model_matrix.fit(D)

    constraints = np.zeros((len(model_matrix.terms),)*2)
    constraints[3,4] = 1
    constraints[4,5] = 1
    validator = validator_from_constraints(model_matrix,
                                           constraints)
                                           
    strategy = min_max(model_matrix,
                       min_terms=1,
                       max_terms=len(model_matrix.terms),
                       lower_terms=['B','C'],
                       validator=validator)

    min_max_selector = FeatureSelector(LinearRegression(),
                                       strategy,
                                       cv=3)

    min_max_selector.fit(D, Y)

    print(min_max_selector.results_)
    print('selected')
    print(min_max_selector.selected_state_)
    print(min_max_selector.results_[min_max_selector.selected_state_])

    
