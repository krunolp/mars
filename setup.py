import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as f:
    requirements = f.readlines()

setuptools.setup(
    name="MARS",
    version="0.0.1",
    author="Krunoslav Lehman Pavasovic, Jonas Rothfuss",
    author_email="krunolp@gmail.com",
    description="MARS: Meta-Learning as Score Matching in the Function Space",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='github.com/krunolp/mars',
    package_dir={'mars': 'mars'},
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'sklearn',
        'tensorflow>2.8.0',
        'tensorflow-probability',
        'matplotlib',
        'jax',
        'jaxlib',
        'GPJax',
    ],
)
