# AutoCL

AutoCL is one AutoML pipeline for concept learning including feature selection and hyperaparameter optimization

- [Installation](#installation)

# Installation

### Installation AutoCL from source

```shell
git clone https://github.com/dice-group/Ontolearn.git](https://github.com/AutoCL2023/AutoCL.git
cd intolearn
conda create --name temp python=3.8
conda activate temp
conda env update --name temp
python -c 'from setuptools import setup; setup()' develop
python -c "import ontolearn"
python -m pytest tests # Partial test with pytest
tox  # full test with tox

```
### Dataset
Our Dataset from SML-Bench (Structured Machine Learning Benchmark) is a benchmark for machine learning from structured data. It provides datasets, which contain structured knowledge (beyond plain feature vectors) in languages such as the Web Ontology Language (OWL) or the logic programming language Prolog. 


