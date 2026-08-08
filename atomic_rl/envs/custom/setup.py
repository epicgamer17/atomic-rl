# TODO: nested setup.py is excluded from the wheel; consider deleting.
from setuptools import setup, find_packages

setup(
    name="custom_gym_envs",
    version="0.0.1",
    packages=find_packages(),
)
