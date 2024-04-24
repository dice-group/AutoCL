# AutoCL

Node classification in knowledge graphs aids in the discovery of new drugs, the identification of risky users in social networks, and the completion of missing type information in knowledge graphs. A crucial requirement for many stakeholders is to understand the models’ predictions in these applications. To this end, concept learners have been proposed to learn concepts in description logic from positive and negative nodes in knowledge graphs in an interpretable way. However, learning concepts on large datasets is computationally expensive. Manually selecting important subgraphs and features to reduce runtime is tedious. While many feature selection approaches have been proposed to simplify the process for tabular data, they are not directly applicable to knowledge graphs and concept learning. In addition, current concept learners have a large number of hyperparameters that need to be optimized to achieve high predictive performance. In this paper, we propose AutoCL—an AutoML approach that is tailored to concept learning. AutoCL comprises methods for automatic feature selection and hyperparameter optimization for concept learners. We demonstrate its effectiveness with SML-Bench, a benchmarking framework for structured machine learning. Our approach leads to better predictive performance across concept learners, in terms of F1-measure and Accuracy while reducing runtime on 6 of 8 datasets from SML-Bench.

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

Our first idea is to use a table-based wrapper method for feature selection: We employeed OwlReady2 to extra the features from KGs. Functions ```get_data_properties```, ```get_object_properties``` are used to get DatatypeProperty features and ObjectProperty features.Then ```transform_data_properties``` and ```transform_object_properties``` are used to covert the features into tabular format.
Finaly, search the best features from ```select_k_best_features``` function based on chi square test scores.

Our second idea is to use a graph-based wrapper method for feature selection: We run EvoLearner, which directly operates on the graph structure, to obtain the relevant top features from concepts via the function ``` get_prominent_properties_occurring_in_top_k_hypothesis ```






