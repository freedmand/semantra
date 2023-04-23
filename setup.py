from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()
with open("requirements.txt", "r") as f:
    requirements = [
        req.strip()
        for req in f.read().splitlines()
        if req.strip()
        and not req.strip().startswith("#")
        and not req.strip().startswith("-")
    ]
with open("VERSION", "r") as f:
    version = f.read().strip()


setup(
    name="semantra",
    version=version,
    description="A semantic search CLI tool",
    long_description=long_description,
    url="https://github.com/freedmand/semantra",
    author="Dylan Freedman",
    author_email="freedmand@gmail.com",
    py_modules=["semantra"],
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.9",
    include_package_data=True,
    package_data={"": ["client/public", "VERSION"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "console_scripts": [
            "semantra = semantra:main",
        ]
    },
)
