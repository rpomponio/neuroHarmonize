Instructions for updating package on PyPI 

https://packaging.python.org/tutorials/packaging-projects/#uploading-your-project-to-pypi

Follow these instructions exactly, with the following changes:

1. Copy contents of /neuroHarmonize/ to /src/neuroHarmonize/
2. Move .setup.cfg to setup.cfg and move setup.py to .setup.py (overrides setup.py)
3. When uploading, run this command:

```python3 -m twine upload dist/*```

