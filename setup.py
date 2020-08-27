import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mofex",
    version="1.0.0",
    author="IISY at Beuth",
    author_email="iisy@beuth-hochschule.de",
    description="Motion Capturing Feature Extraction.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.beuth-hochschule.de/iisy/mofex-mocap-feature-extractor",
    package_dir={'': 'src'},
    packages=setuptools.find_packages(where='src'),
    install_requires=['numpy', 'opencv-python', 'plotly', 'torch', 'torchvision', 'chart-studio'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
