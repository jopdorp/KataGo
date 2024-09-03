from setuptools import setup, Extension
from Cython.Build import cythonize
import glob
import os

# Get all the .py files in the current directory (excluding setup.py)
py_files = glob.glob("*.py")
py_files.remove("setup.py")

# Define extensions
extensions = [
    Extension(os.path.splitext(file)[0], [file]) for file in py_files
]

setup(
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"}),
)