import setuptools

setuptools.setup(
    name = "object_localizer",
    version = "0.0.1",
    author = "A.Korovko",
    description = "localizes main object on a photo",
    packages = setuptools.find_packages(),
    install_requires = [
        "numpy",
        "opencv-python",
        "pandas",
        "torch==0.4.1",
        "torchvision",
        "Pillow",
        "click",
        "tqdm==4.24.0"
    ]
)