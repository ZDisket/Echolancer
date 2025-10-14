from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="echolancer",
    version="1.0.0",
    author="Echolancer Team",
    author_email="echolancer@example.com",
    description="A standalone Transformer-based text-to-speech model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/echolancer/echolancer",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "black>=21.0.0",
            "flake8>=4.0.0",
        ],
        "audio": [
            "librosa>=0.9.0",
        ],
        "extended": [
            "jiwer>=2.3.0",
            "nltk>=3.6",
        ],
    },
    entry_points={
        "console_scripts": [
            "echolancer-train=train:main",
            "echolancer-infer=infer:main",
        ],
    },
)