from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='neuroHarmonize',
      version='2.4.5',
      description='Harmonization tools for multi-center neuroimaging studies.',
      long_description=readme(),
      long_description_content_type='text/markdown',
      url='https://github.com/rpomponio/neuroHarmonize',
      author='Raymond Pomponio',
      author_email='raymond.pomponio@outlook.edu',
      license='MIT',
      packages=['neuroHarmonize'],
      python_requires='>=3.9',
      install_requires=[
          'numpy>=1.19.0,<2.0.0',
          'pandas>=1.1.0,<3.0.0',
          'nibabel>=3.0.0,<6.0.0',
          'statsmodels>=0.12.0,<1.0.0',
          'neuroCombat==0.2.12'
      ],
      extras_require={
          'dev': ['pytest>=7.0.0', 'pytest-cov>=4.0.0']
      },
      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.9',
          'Programming Language :: Python :: 3.10',
          'Programming Language :: Python :: 3.11',
          'Programming Language :: Python :: 3.12',
          'Topic :: Scientific/Engineering :: Medical Science Apps.',
          'Topic :: Scientific/Engineering :: Image Processing',
      ],
      keywords='neuroimaging harmonization combat multi-site',
      project_urls={
          'Bug Reports': 'https://github.com/rpomponio/neuroHarmonize/issues',
          'Source': 'https://github.com/rpomponio/neuroHarmonize',
      },
      zip_safe=False)
