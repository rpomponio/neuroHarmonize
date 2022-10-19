Instructions for updating package on PyPI 

https://packaging.python.org/tutorials/packaging-projects/#uploading-your-project-to-pypi


With the new `pyproject.toml`, these are the steps:

First, make sure you to install the dependencies:

```
pip install --upgrade twine build
```

Follow these instructions exactly, with the following changes:

1. Build the project:

```
python3 -m build
```

2. Upload to pypi:

```
python3 -m twine upload dist/*
```

