from setuptools import setup, find_packages

setup(
    name="qnn_verifier",
    version="0.1.0",
    description="Lasserre Hierarchy based Quantized Neural Network Verification",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "cvxpy>=1.3.0",
        "sympy>=1.12",
        "networkx>=3.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
    ],
)
