from setuptools import setup, find_packages

REQUIREMENTS = ["fenics-dolfinx>=0.6.0"]

setup(
    name="adFEM",
    description="Adaptive finite element algorithms for linear elasticity.",
    author="Maximilian Brodbeck",
    author_email="maximilian.brodbeck@isd.uni-stuttgart.de",
    python_requires=">=3.10.12",
    packages=find_packages(),
    install_requires=REQUIREMENTS,
    zip_safe=False,
)
