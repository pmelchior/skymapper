from setuptools import setup

long_description = open('README.md').read()

setup(
    name="skymapper",
    description="Mapping astronomical survey data on the sky, handsomely",
    long_description=long_description,
    long_description_content_type='text/markdown',
    version="0.4.5",
    license="MIT",
    author="Peter Melchior",
    author_email="peter.m.melchior@gmail.com",
    url="https://github.com/pmelchior/skymapper",
    packages=["skymapper", "skymapper.survey"],
    package_data={"skymapper.survey": ['*.ply']},
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: Astronomy"
    ],
    keywords = ['visualization','map projection','matplotlib'],
    install_requires=["matplotlib", "numpy", "scipy", "healpy"]
)
