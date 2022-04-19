from setuptools import setup, find_packages

with open("README.md", mode="r", encoding="utf-8") as readme_file:
    readme = readme_file.read()

setup(
    name="imagesearch",
    version="1.0.0",
    author="Nandan Thakur",
    author_email="nandant@gmail.com",
    description="",
    long_description=readme,
    long_description_content_type="text/markdown",
    license="Apache License 2.0",
    url="https://github.com/LachlanGray/Image-Search",
    download_url="",
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=[
        'sklearn',
        'torch'
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ]
)