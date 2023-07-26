import os, nbformat
from glob import glob

for f in glob('source/labs/Ch14*'):
    os.remove(f)



for nbfile in glob('source/labs/*nb'):
    base = os.path.splitext(nbfile)[0]
    fname = os.path.split(base)[1]

    colab_code = f'''
<a target="_blank" href="https://colab.research.google.com/github/intro-stat-learning/ISLP/main/main/docs/labs/{fname}.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>

</a>
'''

    # allow errors for lab2
    if fname[:3] == 'Ch6':
        colab_code = ('''
```{code-cell}
:tags: [hide-cell]
        
import warnings
warnings.simplefilter('ignore')        
```        

''' + colab_code)
    if fname[:3] == 'Ch2':
        nb = nbformat.read(open(nbfile), 4)
        nb.metadata.setdefault('execution', {})['allow_errors'] = True
        nbformat.write(nb, open(nbfile, 'w'))

    if fname[:4] != 'Ch10':

        os.system(f'jupytext --set-formats ipynb,md:myst {base}.ipynb; jupytext --sync {base}.ipynb')

        myst = open(f'{base}.md').read().strip()

        new_myst = []
        for l in myst.split('\n'):
            if l.strip()[:9] != '# Chapter':
                new_myst.append(l)
        new_myst.append(colab_code)
        
        myst = '\n'.join(new_myst) # remove the "Chapter %d
        myst = myst.replace('# Lab:', colab_code + '\n# ')

        open(f'{base}.md', 'w').write(myst)

        os.system(f'jupytext --sync {base}.ipynb')

# add a warning for ridge
# at ## Ridge Regression and the Lasso

code_to_insert = '''

<!-- #region -->
```{attention}
Using `skl.ElasticNet` to fit ridge regression
throws up many warnings. We have inserted the code below to filter them.
```
<!-- #endregion -->

```{code-cell}
import warnings
warnings.simplefilter('ignore')
```

'''

