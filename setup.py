from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='neuroHarmonize',
      version='1.0.0',
      description='Harmonization tools for multi-center neuroimaging studies.',
      long_description=readme(),
      url='https://github.com/rpomponio/neuroHarmonize',
      author='Raymond Pomponio',
      author_email='raymond.pomponio@outlook.edu',
      license='MIT',
      packages=['neuroHarmonize'],
      install_requires=['numpy', 'pandas', 'nibabel', 'statsmodels>=0.11.0.dev0'],
      zip_safe=False)
