import os, nbformat
from glob import glob

for f in glob('source/labs/Ch14*'):
    os.remove(f)



for nbfile in glob('source/labs/*nb'):
    base = os.path.splitext(nbfile)[0]
    labname = os.path.split(base)[1]

    colab_code = f'''
<a target="_blank" href="https://colab.research.google.com/github/intro-stat-learning/ISLP/blob/main/docs/source/labs/{labname}.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>

</a>
'''

    # allow errors for lab2
    if labname[:3] == 'Ch6':
        colab_code = ('''
```{code-cell}
:tags: [hide-cell]
        
import warnings
warnings.simplefilter('ignore')        
```        

```{attention}
Using `skl.ElasticNet` to fit ridge regression
throws up many warnings. We have suppressed them below.
```

''' + colab_code)
    if labname[:3] == 'Ch2':
        nb = nbformat.read(open(nbfile), 4)
        nb.metadata.setdefault('execution', {})['allow_errors'] = True
        nbformat.write(nb, open(nbfile, 'w'))

    if labname[:4] != 'Ch10':

        os.system(f'jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace {nbfile}')

        os.system(f'jupytext --set-formats ipynb,md:myst {nbfile}; jupytext --sync {nbfile}')

        myst = open(f'{base}.md').read().strip()

        new_myst = []
        for l in myst.split('\n'):
            if l.strip()[:9] != '# Chapter':
                if 'Lab:' in l:
                    l = '# ' + l[6:] + '\n' + colab_code
                new_myst.append(l)
            
        myst = '\n'.join(new_myst) # remove the "Chapter %d

        open(f'{base}.md', 'w').write(myst)

        os.system(f'jupytext --sync {base}.ipynb; rm {base}.md')

        cmd = f'jupyter nbconvert --execute --inplace {nbfile}'
        if labname[:3] == 'Ch2':
            cmd += ' --allow-errors'
        os.system(cmd)


for nbfile in glob('source/helpers/*nb') + glob('source/datasets/*nb'):
    cmd = f'jupyter nbconvert --execute --inplace {nbfile}'
    os.system(cmd)


# add a warning for ridge
# at ## Ridge Regression and the Lasso

