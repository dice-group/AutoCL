# AutoCL

Node classification in Knowledge Graphs (KGs) aids in the discovery of new drugs, the identification of risky users in social networks, and the completion of missing type information in KGs. As many stakeholders in these applications need to understand the models' predictions, concept learners have been proposed to learn concepts in description logics from positive and negative nodes in knowledge graphs. However, as dataset sizes increase, so does the computational time to learn concepts, and data scientists need to spend a lot of time finetuning hyperparameters. While many AutoML approaches have been proposed to simplify the datascience process for tabular data, none of these approaches is directly applicable to graph data. In this paper, we propose AutoCL---an AutoML approach that is tailored to knowledge graphs and concept learning. It provides methods for automatic feature selection in knowledge graphs and hyperparameter optimization of concept learners. We demonstrate its effectiveness with SML-Bench, a benchmarking framework for structured machine learning. Our feature selection improves the runtime of concept learners while maintaining predictive performance in terms of $F_1$-measure and $Accuracy$ and our hyperparameter optimization leads to better predictive performance on many datasets.


<img width="614" alt="onto" src="https://user-images.githubusercontent.com/123487952/215816088-242fbf1e-3cb8-4956-b65b-8bfa1c34868f.png">


- [Installation](#installation)

# Installation

### Installation AutoCL from source

```shell
git clone https://github.com/AutoCL2023/AutoCL.git
cd AutoCL
conda create --name temp python=3.8
conda activate temp
conda install -c conda-forge optuna=3.0.3
conda install -c conda-forge owlready2=0.41
conda install scikit-learn=1.0.2
conda env update --name temp
python -c 'from setuptools import setup; setup()' develop
python -c "import ontolearn"
```
### Dataset
Our Dataset from SML-Bench (Structured Machine Learning Benchmark) is a benchmark for machine learning from structured data. It provides datasets, which contain structured knowledge (beyond plain feature vectors) in languages such as the Web Ontology Language (OWL) or the logic programming language Prolog. 

# Our contributions

### Hyperparameter Optimization(HPO)
We proposed one approach for hyperparameter optimization of concept learners based one Covariance Matrix Adaptation Evolution Strategy(CMA-ES) sampler from Optuna framework.
Our HPO source code shown in ``` hyperparameter optimization approach ``` from ``` AutoCL\examples ```.
For each concept learner, We ran CMA-ES sampler via ``` get_best_optimization_result_for_cmes_sampler ``` function in order to obtain the best hyperparameters of our concepet learner.


### Feature Selection
We provides methods for automatic feature selection in knowledge graphs.
Our two Feature Selection appraoches source code shown in ``` table-based feature selection ``` and  ``` graph-based feature selection ```  from ``` AutoCL\examples\feature selection approach ``` folder.

Our first idea is to use a table-based wrapper method for feature selection:
Our second idea is to use a graph-based wrapper method for feature selection: We run EvoLearner, which directly operates on the graph structure, to obtain the relevant top features from concepts via the function ``` get_prominent_properties_occurring_in_top_k_hypothesis ```


The raw data was saved in the KGs,we employeed OwlReady2 to extra the features from KGs. Functions ```get_data_properties```, ```get_object_properties``` are used to get DatatypeProperty features and ObjectProperty features.Then ```transform_data_properties``` and ```transform_object_properties``` are used to covert the features into tabular format.
Finaly, search the best features from ```select_k_best_features``` function based on chi square test scores.



