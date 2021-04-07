from setuptools import find_packages, setup

setup(
    name='postImageClassification',
    extras_require=dict(tests=['pytest']),
    packages=find_packages('src'),
    package_dir={"":"src"}
)