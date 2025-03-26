from setuptools import setup, find_packages

setup(
    name="qcongen",
    version="0.1.0",
    description="Quantum Constraint Generation Algorithm",
    author="András Czégel",  
    author_email="czegel@inf.u-szeged.hu", 
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.11",
    install_requires=[
        "numpy>=2.2.2",
    ],
    extras_require={
        "dev": [
            "black>=25.1.0",
            "ruff>=0.9.4",
            "mypy>=1.14.1",
        ],
    },
) 