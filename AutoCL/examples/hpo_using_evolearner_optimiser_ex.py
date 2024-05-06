import json
import os
import numpy as np
from optuna.samplers import RandomSampler

from ontolearn.knowledge_base import KnowledgeBase
from owlapy.model import OWLNamedIndividual, IRI
from random import shuffle
from ontolearn.learning_problem import PosNegLPStandard
from AutoCL.hpo.optimisers import EvolearnerOptimizer


if __name__ == "__main__":
    DATASET = 'carcinogenesis_lp'
    current_dir = os.getcwd()
    print('Current', current_dir)
    path = f'../dataset/{DATASET}.json'
    print('path', path)
    with open(path) as json_file:
        settings = json.load(json_file)

    kb = KnowledgeBase(path=settings['data_path'])
    for str_target_concept, examples in settings['problems'].items():

        p = set(examples['positive_examples'])
        n = set(examples['negative_examples'])

        typed_pos = list(set(map(OWLNamedIndividual, map(IRI.create, p))))
        typed_neg = list(set(map(OWLNamedIndividual, map(IRI.create, n))))

        # Shuffle the Positive and Negative Sample
        shuffle(typed_pos)
        shuffle(typed_neg)

        # Split the data into Training Set, Validation Set and Test Set
        train_pos, val_pos, test_pos = np.split(typed_pos,
                                                [int(len(typed_pos) * 0.6),
                                                 int(len(typed_pos) * 0.8)]
                                                )
        train_neg, val_neg, test_neg = np.split(typed_neg,
                                                [int(len(typed_neg) * 0.6),
                                                 int(len(typed_neg) * 0.8)]
                                                )
        train_pos, train_neg = set(train_pos), set(train_neg)
        val_pos, val_neg = set(val_pos), set(val_neg)
        test_pos, test_neg = set(test_pos), set(test_neg)
        lp = PosNegLPStandard(pos=train_pos, neg=train_neg)
        # Create optimizer object
        optimizer = EvolearnerOptimizer(settings['data_path'], kb, lp, str_target_concept, val_pos, val_neg)

        # Set sampler
        optimizer.set_sampler(RandomSampler())
        # Optimize using categorical distribution
        optimizer.optimize(optimizer.objective_with_categorical_distribution, n_trials=1)

        # Save results within the EvolearnerOptimizer class
        optimizer.save_results()

        # Get the best hpo
        best_hpo = optimizer.get_best_hpo()
