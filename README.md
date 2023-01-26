# AutoCL
AutoCL: AutoML for Concept Learning

Abstract:
Node classification in Knowledge Graphs (KGs)
aids in the discovery of new drugs, the identifi-
cation of risky users in social networks, and the
completion of missing type information in KGs.
As many stakeholders in these applications need
to understand the models’ predictions, concept
learners have been proposed to learn concepts
in description logics from positive and negative
nodes in knowledge graphs. However, as dataset
sizes increase, so does the computational time to
learn concetps, and data scientists need to spend
a lot of time finetuning hyperparameters. While
many AutoML approaches have been proposed
to simplify the datascience process for tabular
data, none of these approaches is directly appli-
cable to graph data. In this paper, we propose
AutoCL—an AutoML approach that is tailored
to knowledge graphs and concept learning. It
provides methods for automatic feature selection
in knowledge graphs and hyperparameter opti-
mization of concept learners. We demonstrate
its effectiveness with SML-Bench, a benchmark-
ing framework for structured machine learning.
Our feature selection improves the runtime of con-
cept learners while maintaining predictive perfor-
mance in terms of F1-measure and accuracy and
our hyperparameter optimization leads to better
predictive performance on many dataset
