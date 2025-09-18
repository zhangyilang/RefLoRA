import setuptools

with open("readme.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="reflora",
    version="0.0.1",
    author="Yilang Zhang",
    author_email="zhan7453@umn.edu",
    description="RefLoRA: Refactored Low-Rank Adaptation for Efficient Fine-Tuning of Large Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zhangyilang/RefLoRA",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
