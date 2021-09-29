import numpy as np, pandas as pd
from sklearn.base import clone

from ...transforms import Poly, NaturalSpline, BSpline, Interaction
from ..model_matrix import ModelMatrix, Variable

from sklearn.preprocessing import (OneHotEncoder,
                                   OrdinalEncoder)
default_encoders = {'categorical': OneHotEncoder(drop=None, sparse=False),
                    'ordinal': OrdinalEncoder()}

def test_interaction():

    I = Interaction(['V', 'U'],
                    {'V':[0,2],
                     'U':[1,3,5]})
    X = np.random.standard_normal((50,10))
    W = I.fit_transform(X)

    W2 = np.array([X[:,0]*X[:,1],
                   X[:,0]*X[:,3],
                   X[:,0]*X[:,5],
                   X[:,2]*X[:,1],
                   X[:,2]*X[:,3],
                   X[:,2]*X[:,5]]).T
    print(np.linalg.norm(W-W2))
    print(W.columns)

def test_ndarray():
    
    X = np.random.standard_normal((50,5))

    M = ModelMatrix(terms=[1, (3,2)],
                    default_encoders=default_encoders)
    M.fit(X)
    MX = M.transform(X)

    np.testing.assert_allclose(X[:,1], MX[:,1])
    np.testing.assert_allclose(X[:,2] * X[:,3], MX[:,2])
    
def test_dataframe1():
    
    X = np.random.standard_normal((50,5))
    D = pd.DataFrame(X, columns=['A','B','C','D','E'])
    
    M = ModelMatrix(terms=['A','D',('D','E')],
                    default_encoders=default_encoders)
    clone(M)
    MX = np.asarray(M.fit_transform(D))

    np.testing.assert_allclose(X[:,0], MX[:,1])
    np.testing.assert_allclose(X[:,3], MX[:,2])
    np.testing.assert_allclose(X[:,3]*X[:,4], MX[:,3])    

def test_dataframe2():
    
    X = np.random.standard_normal((50,5))
    D = pd.DataFrame(X, columns=['V','B','A','D','E'])
    
    M = ModelMatrix(terms=['A', 'D', 'B', ('D','E'), 'V'],
                    default_encoders=default_encoders)
    clone(M)

    MX = M.fit_transform(D)

    X2 = D.copy()
    MX2 = M.transform(D)
    np.testing.assert_allclose(MX, MX2)
    
def test_dataframe3():
    
    X = np.random.standard_normal((50,5))
    D = pd.DataFrame(X, columns=['A','B','C','D','E'])
    D['E'] = pd.Categorical(np.random.choice(range(4,8), 50, replace=True))
    
    M = ModelMatrix(terms=['A', 'E', ('D','E')],
                    default_encoders=default_encoders)
    MX = np.asarray(M.fit_transform(D))
    M2 = clone(M)

    DE = pd.get_dummies(D['E'])
    np.testing.assert_allclose(X[:,0], MX[:,1])
    np.testing.assert_allclose(DE, MX[:,2:6])    
    np.testing.assert_allclose(np.asarray(DE) * np.multiply.outer(X[:,3], np.ones(4)), MX[:,-4:])

    X2 = D.copy()
    MX2 = M.transform(D)
    np.testing.assert_allclose(MX, MX2)

def test_dataframe4():
    
    X = np.random.standard_normal((50,5))
    D = pd.DataFrame(X, columns=['A','B','C','D','E'])
    D['D'] = pd.Categorical(np.random.choice(['a','b','c'], 50, replace=True))
    D['E'] = pd.Categorical(np.random.choice(range(4,8), 50, replace=True))
    
    M = ModelMatrix(terms=['A', 'E', ('D','E')],
                    default_encoders=default_encoders)
    MX = np.asarray(M.fit_transform(D))

    DE = pd.get_dummies(D['E'])
    np.testing.assert_allclose(X[:,0], MX[:,1])
    np.testing.assert_allclose(DE, MX[:,2:6])    

    # check they agree on copy of dataframe

    X2 = D.copy()
    MX2 = M.transform(D)
    np.testing.assert_allclose(MX, MX2)

def test_dataframe5():
    
    X = np.random.standard_normal((50,5))
    D = pd.DataFrame(X, columns=['A','B','C','D','E'])
    D['D'] = pd.Categorical(np.random.choice(['a','b','c'], 50, replace=True))
    D['E'] = pd.Categorical(np.random.choice(range(4,8), 50, replace=True))
    
    M = ModelMatrix(terms=['A', 'E', ('D','E')],
                    default_encoders=default_encoders)
    MX = np.asarray(M.fit_transform(D))

    # check they agree on copy of dataframe

    X2 = D.copy()
    MX2 = M.transform(D)
    np.testing.assert_allclose(MX, MX2)
    
def test_dataframe6():
    
    X = np.random.standard_normal((50,5))
    D = pd.DataFrame(X, columns=['A','B','C','D','E'])
    W = Variable(('A','E'), 'AE', None)
    D['D'] = pd.Categorical(np.random.choice(['a','b','c'], 50, replace=True))
    D['E'] = pd.Categorical(np.random.choice(range(4,8), 50, replace=True))
    
    M = ModelMatrix(terms=['A',W,(W,'D',)],
                    default_encoders=default_encoders)
    MX = M.fit_transform(D)

    MX = np.asarray(MX)

def test_dataframe7():
    
    X = np.random.standard_normal((50,6))
    D = pd.DataFrame(X, columns=['AA','Bbbb','C','Ddd','Y','Eee'])
    D['Ddd'] = pd.Categorical(np.random.choice(['a','b','c'], 50, replace=True))
    D['Eee'] = pd.Categorical(np.random.choice(range(4,8), 50, replace=True))
        
    M = ModelMatrix(terms=D.columns.drop(['Y','C']),
                    default_encoders=default_encoders)
    MX = M.fit_transform(D)
    print(MX.columns)
    MX = np.asarray(MX)

def test_dataframe8():
    
    X = np.random.standard_normal((50,6))
    D = pd.DataFrame(X, columns=['A','B','C','D','Y','E'])
    D['D'] = pd.Categorical(np.random.choice(['a','b','c'], 50, replace=True))
    D['E'] = pd.Categorical(np.random.choice(range(4,8), 50, replace=True))
    
    poly =  Poly(degree=3)
    # raises a ValueError because poly will have been already fit -- need new instance of Poly
    W = Variable(('A',), 'poly(A)', poly)
    M = ModelMatrix(terms=list(D.columns.drop(['Y','C'])) + [(W,'E')],
                    default_encoders=default_encoders)
    MX = M.fit_transform(D)

    print(MX.columns)
    MX = np.asarray(MX)

def test_dataframe9():
    
    X = np.random.standard_normal((50,6))
    D = pd.DataFrame(X, columns=['A','B','C','D','Y','E'])
    D['D'] = pd.Categorical(np.random.choice(['a','b','c'], 50, replace=True))
    D['E'] = pd.Categorical(np.random.choice(range(4,8), 50, replace=True))
    
    poly =  Poly(degree=3)
    # raises a ValueError because poly will have been already fit -- need new instance of Poly
    W = Variable(('A',), 'poly(A)', poly)
    U = Variable(('B',), 'poly(B)', clone(poly))
    M = ModelMatrix(terms=list(D.columns.drop(['Y','C'])) + [W,U],
                    default_encoders=default_encoders)
    MX = M.fit_transform(D)

    print(MX.columns)
    MX = np.asarray(MX)

def test_dataframe10():
    
    X = np.random.standard_normal((50,5))
    D = pd.DataFrame(X, columns=['A','B','C','D','E'])
    W = Variable(('A','E'), 'AE', None)
    U = Variable((W, 'C'), 'WC', None)
    D['D'] = pd.Categorical(np.random.choice(['a','b','c'], 50, replace=True))
    D['E'] = pd.Categorical(np.random.choice(range(4,8), 50, replace=True))
    
    M = ModelMatrix(terms=['A', 'E', 'C', W, (W, 'D',), U],
                    default_encoders=default_encoders)
    MX = M.fit_transform(D)
    print(MX.columns)
    MX = np.asarray(MX)

    V = MX[:,-6:]
    V2 = np.column_stack([MX[:,M.column_map_['A']],
                          MX[:,M.column_map_['E']],
                          MX[:,M.column_map_['C']]])
    print(np.linalg.norm(V-V2))
