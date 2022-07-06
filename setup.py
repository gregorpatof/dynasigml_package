import os.path
import setuptools

HERE = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(HERE, "README.md")) as f:
    README = f.read()

setuptools.setup(
    name="dynasigml",
    version="1.0.0",
    author="Olivier Mailhot",
    description="DynaSig-ML",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/gregorpatof/dynasigml_package",
    packages=setuptools.find_packages('src'),
    package_dir={'':'src'},

    install_requires=['numpy', 'nrgten>=1.1.14', 'sklearn'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)