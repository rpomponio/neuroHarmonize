from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='neuroHarmonize',
      version='2.4.0',
      description='Harmonization tools for multi-center neuroimaging studies.',
      long_description=readme(),
      url='https://github.com/rpomponio/neuroHarmonize',
      author='Raymond Pomponio',
      author_email='raymond.pomponio@outlook.edu',
      license='MIT',
      packages=['neuroHarmonize'],
      install_requires=['numpy', 'pandas', 'nibabel', 'statsmodels>=0.12.0', 'neuroCombat==0.2.12'],
      zip_safe=False)
