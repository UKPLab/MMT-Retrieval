from setuptools import setup, find_packages

with open("README.md", mode="r", encoding="utf-8") as readme_file:
    readme = readme_file.read()

setup(
    name="mmt-retrieval",
    version="0.1.1",
    author="Gregor Geigle",
    author_email="gregor.geigle@gmail.com",
    description="Multimodal Transformers for Image-Text Retrieval and more",
    long_description=readme,
    long_description_content_type="text/markdown",
    license="MIT",
    url="https://github.com/UKPLab/MMT-Retrieval",
    packages=find_packages(),
    install_requires=[
        "sentence_transformers>=0.4.1.2",
        "tqdm>=4.32.1",
        "requests>=2.22.0",
        "transformers>=4.1.1",
        "numpy>=1.19.3",
        "torch>=1.6.0"

    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    keywords="Transformer Networks Oscar UNITER M3P multimodal embedding PyTorch NLP CV deep learning image search retrieval"
)