import os, nbformat
from glob import glob

import __main__
dirname = os.path.split(__main__.__file__)[0]
print(dirname)

for f in glob(os.path.join(dirname, 'source', 'labs', 'Ch14*')):
    os.remove(f)
    print(f)
    
version = 'v2'
main = 'main'

print(f'checking out version {version} of the labs')
print(f'checking out version {main} of ISLP')

os.system(f'''
cd {dirname}/ISLP_labs;
git checkout {version};
mkdir -p {dirname}/source/labs;
cp -r * {dirname}/source/labs;
git checkout {main};
''')

for nbfile in glob(os.path.join(dirname, 'source', 'labs', '*nb')):
    base = os.path.splitext(nbfile)[0]
    labname = os.path.split(base)[1]

    colab_code = f'''
<a target="_blank" href="https://colab.research.google.com/github/intro-stat-learning/ISLP_labs/blob/{version}/{labname}.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>

</a>

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/intro-stat-learning/ISLP_labs/{version}?labpath={labname}.ipynb)

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

        # clear outputs for all but Ch10
        os.system(f'jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace {nbfile}')

    os.system(f'jupytext --set-formats ipynb,md:myst {nbfile}; jupytext --sync {nbfile}')

    myst = open(f'{base}.md').read().strip()

    new_myst = []
    for l in myst.split('\n'):
        if l.strip()[:9] != '# Chapter':
            if 'Lab:' in l:
                l = l.replace('Lab:', '') + '\n' + colab_code
            new_myst.append(l)

    myst = '\n'.join(new_myst) # remove the "Chapter %d

    open(f'{base}.md', 'w').write(myst)

    os.system(f'jupytext --sync {base}.ipynb; rm {base}.md')

