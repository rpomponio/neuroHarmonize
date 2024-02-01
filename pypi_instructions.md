Instructions for updating package on PyPI.

See the following link for detailed instructions, though note that several steps are different.

<https://packaging.python.org/tutorials/packaging-projects/#uploading-your-project-to-pypi>

Follow the instructions in the link, except prepare the repository for distribution:

1. Upgrade packages `pip`, `build`, and `twine`.
2. Copy contents of `/neuroHarmonize/` to `/src/neuroHarmonize/`.

      ```cp neuroHarmonize/* src/neuroHarmonize/```

3. Move .setup.cfg to setup.cfg and move setup.py to .setup.py (this overrides setup.py).

      ```mv .setup.cfg setup.cfg```
      ```mv setup.py .setup.py```

4. Build the package.

      ```python3 -m build ```

5. Upload the distribution. Note the username will be `__token__` and the password will be the API token from <https://pypi.org>.

      ```python3 -m twine upload dist/*```

6. Finally (optional) discard all local changes in the repository to revert to developer state.

      ```git stash```
