from setuptools import setup, find_packages

setup(
    name="wapigt",
    version="0.1.0",
    description="Wavelet-Packet Physics-Informed Graph Transformer for bearing fault diagnosis",
    author="Research Team",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torch-geometric>=2.3.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "pandas>=1.3.0",
        "PyWavelets>=1.1.0",
        "PyYAML>=6.0",
        "optuna>=3.0.0",
        "tensorboard>=2.10.0",
        "tqdm>=4.62.0",
        "rich>=10.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "h5py>=3.0.0",
        "mat73>=0.56",
        "pytest>=7.0.0",
        "pytest-cov>=3.0.0",
    ],
    entry_points={
        "console_scripts": [
            "wapigt-train=scripts.train_wapigt:main",
        ],
    },
)
