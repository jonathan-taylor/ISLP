import matplotlib.pyplot as plt
import numpy as np

def plot_svm(X,
             y,
             svm, # we assume that (X,y) has been passed to `fit` for `svm`
             xlim=None,
             ylim=None,
             ax=None,
             decision_cmap=plt.cm.plasma,
             scatter_cmap=plt.cm.tab10,
             nx=300,
             ny=300,
             alpha=0.2):

   if xlim is None:
      xlim = (X[:,0].min()-0.5*X[:,0].std(),X[:,0].max()+0.5*X[:,0].std())

   if ylim is None:
      ylim = (X[:,1].min()-0.5*X[:,1].std(),X[:,1].max()+0.5*X[:,1].std())

   if ax is None:
      fig, ax = plt.subplots()
   else:
      fig = ax.figure

   # draw the points

   ax.scatter(X[:,0], X[:,1], c=y, cmap=scatter_cmap)

   # add the contour

   xval, yval = np.meshgrid(np.linspace(xlim[0], xlim[1], nx),
                            np.linspace(ylim[0], ylim[1], ny))   

   # this will work well when labels are integers

   prediction_val = svm.predict(np.array([xval.reshape(-1),
                                          yval.reshape(-1)]).T)
   ax.contourf(xval,
               yval,
               prediction_val.reshape(yval.shape),
               cmap=decision_cmap,
               alpha=alpha)

   # add the support vectors    

   ax.scatter(X[svm.support_,0], 
              X[svm.support_,1], marker='+', c='k', s=200)

