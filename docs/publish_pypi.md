# Publishing new versions of Semantra to PyPI

Note: these instructions are for developers of Semantra that have credentials to publish to PyPI.

- Make coding changes as necessary
- Open `pyproject.toml` and increment the version number
- In a virtual environment, run `pip install -e .`
- Verify the new version is installed via `semantra --version`. Also verify any new changes are working as intended
- Run `rm -rf dist && python3 -m build && python3 -m twine upload dist/*` to build the package and upload to PyPI
- When prompted, enter login credentials
