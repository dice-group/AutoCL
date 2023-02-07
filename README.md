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
conda install -c conda-forge optuna=3.0.3
conda install -c conda-forge owlready2=0.39
conda install scikit-learn=1.0.2
conda env update --name temp
python -c 'from setuptools import setup; setup()' develop
python -c "import ontolearn"
python -m pytest tests # Partial test with pytest
tox  # full test with tox

```
### Dataset
Our Dataset from SML-Bench (Structured Machine Learning Benchmark) is a benchmark for machine learning from structured data. It provides datasets, which contain structured knowledge (beyond plain feature vectors) in languages such as the Web Ontology Language (OWL) or the logic programming language Prolog. 

# Our contributions

### Hyperparameter Optimization(HPO)
We proposed one approach for hyperparameter optimization of concept learners based one TPE sampler from Optuna framework.
Our HPO source code shown in optuna_random_sampler.py from examples folder.
We ran TPE sampler via get_best_optimization_result_for_tpe_sampler function in order to obtain the best hyperparameters of our concepet learner.


### Feature Selection
We provides methods for automatic feature selection in knowledge graphs.
Our Feature Selection source code shown in evolearner_feature_selection.py from examples folder.
The raw data was saved in the KGs,we employeed OwlReady2 to extra the features from KGs. Functions get_data_properties, get_object_properties are used to get DatatypeProperty features and ObjectProperty features.Then transform_data_properties and transform_object_properties are used to covert the features into tabular format.
Finaly, variance-filtered data via calc_variance_threshold function, search the best features from select_k_best_features function.



