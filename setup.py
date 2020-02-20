import sys
from setuptools import setup

assert sys.version_info.major == 3 and sys.version_info.minor >= 7, \
    "The rltorch repo is designed to work with Python 3.7 and greater." \
    + "Please install it before proceeding."

setup(name="rltorch",
      version="0.0.1",
      install_requires=[
          'gym',
          'numpy',
          'Pillow',
          'matplotlib',
          'torch>=1.4.0',
          'torchvision',
          'prettytable'
      ],
      description="Deep RL algorithm implementations using pytorch.",
      author="Jonathon Schwartz",
      )
