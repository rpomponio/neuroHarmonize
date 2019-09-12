from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='neuroHarmonize',
      version='0.1',
      description='Harmonization tools for multi-center neuroimaging studies.',
      long_description=readme(),
      url='https://github.com/rpomponio/neuroHarmonize',
      author='Ray Pomponio',
      author_email='raymond.pomponio@pennmedicine.upenn.edu',
      license='MIT',
      packages=['neuroHarmonize'],
      install_requires=['neuroCombat', 'numpy', 'pandas', 'nibabel'],
      dependency_links=['http://github.com/ncullen93/neuroCombat'],
      zip_safe=False)