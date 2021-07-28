import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="poisson_CNN",
    version="0.3.0",
    author="Ali Girayhan Ozbay",
    author_email="aligirayhan.ozbay14@imperial.ac.uk",
    description="A convolutional neural network based Poisson solver",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aligirayhanozbay/poisson_CNN",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering"
    ],
    install_requires=[
        "tensorflow",
	"tensorflow-probability",
	"pyamg",
	"scipy"    
    ],
    python_requires='>=3.6'
)
