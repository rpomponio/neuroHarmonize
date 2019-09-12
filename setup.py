from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='neuroharmonize',
      version='0.1',
      description='Harmonization tools for multi-center neuroimaging studies.',
      long_description=readme(),
      url='https://github.com/rpomponio/neuroharmonize',
      author='Ray Pomponio',
      author_email='raymond.pomponio@pennmedicine.upenn.edu',
      license='MIT',
      packages=['neuroharmonize'],
      install_requires=['neuroCombat', 'numpy', 'pandas', 'nibabel'],
      dependency_links=['http://github.com/ncullen93/neuroCombat'],
      zip_safe=False)