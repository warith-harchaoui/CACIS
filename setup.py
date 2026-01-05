from setuptools import setup, find_packages
from pathlib import Path

this_dir = Path(__file__).parent

readme = (this_dir / "README.md").read_text(encoding="utf-8")

setup(
    name="cacis",
    version="0.1.0",
    description="Cost-Aware Classification with Implicit Scores",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Warith Harchaoui",
    author_email="wharchaoui@nexton-group.com",
    url="https://github.com/warith-harchaoui/cacis",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0",
        "torchvision",
        "numpy",
        "fasttext",
    ],
)
