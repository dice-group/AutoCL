import json
import os

from ontolearn.knowledge_base import KnowledgeBase
from owlapy.model import OWLNamedIndividual, IRI
from random import shuffle
from ontolearn.learning_problem import PosNegLPStandard

from AutoCL.feature_selection.feature_selector import EvolearnerBasedFeatureSelection as evo
from AutoCL.feature_selection.feature_selector import TableBasedFeatureSelection as tb


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

        # Split the data into Training Set and Test Set
        train_pos = set(typed_pos[:int(len(typed_pos) * 0.8)])
        train_neg = set(typed_neg[:int(len(typed_neg) * 0.8)])
        test_pos = set(typed_pos[-int(len(typed_pos) * 0.2):])
        test_neg = set(typed_neg[-int(len(typed_neg) * 0.2):])
        lp = PosNegLPStandard(pos=train_pos, neg=train_neg)

        # Graph Based
        feature_selector = evo(settings['data_path'], lp, kb, str_target_concept, min_prop=10, n=10)
        # Access the features and new_kb
        features = feature_selector.features
        new_kb = feature_selector.new_kb
        print('features', features)
        print('new_kb', new_kb)

        # Table Based
        feature_selector_tb = tb(settings['data_path'], lp, kb, str_target_concept,p, n)
        features = feature_selector_tb.features
        new_kb = feature_selector_tb.new_kb
        print('features', features)
        print('new_kb', new_kb)
