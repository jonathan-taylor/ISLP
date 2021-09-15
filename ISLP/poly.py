import numpy as np, pandas as pd

def poly(Xseries, degree):
    """
    Create columns of design matrix
    for orthogonal polynomial for a given
    variable    
    """
    result = np.zeros((Xseries.shape[0], degree))
    powX = np.power.outer(Xseries.values, np.arange(1, degree+1))
    powX -= powX.mean(0)
    result[:,0] = powX[:,0] / np.linalg.norm(powX[:,0])
    for i in range(1, degree):
        result[:,i] = powX[:,i]
        for j in range(i):
            result[:,i] -= (result[:,i] * result[:,j]).sum() * result[:,j] 
        result[:,i] /= np.linalg.norm(result[:,i])
    df = pd.DataFrame(result, columns=['poly(%s, %d)' % (Xseries.name, degree) for degree in range(1, degree+1)])
    return df


