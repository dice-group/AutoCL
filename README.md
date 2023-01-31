# AutoCL

AutoCL is one AutoML pipeline for concept learning including feature selection and hyperaparameter optimization
<img width="614" alt="onto" src="https://user-images.githubusercontent.com/123487952/215816088-242fbf1e-3cb8-4956-b65b-8bfa1c34868f.png">


- [Installation](#installation)

# Installation

### Installation AutoCL from source

```shell
git clone https://github.com/AutoCL2023/AutoCL.git
cd AutoCL
cd ontology
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


