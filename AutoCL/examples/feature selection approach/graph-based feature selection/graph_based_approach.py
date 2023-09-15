import json
import os
import time
from random import shuffle
from owlready2 import get_ontology, destroy_entity
import logging

from search import calc_prediction
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.learning_problem import PosNegLPStandard
from owlapy.model import OWLNamedIndividual, IRI
from ontolearn.concept_learner import EvoLearner
from owlapy.render import DLSyntaxObjectRenderer

DIRECTORY = './FS_Evolearner_Evolearner/'
LOG_FILE = 'featureSelectionWithGraph.log'
DATASET = 'mammograph'

path_dataset = f'AutoCL/examples/dataset/{DATASET}.json'

with open(path_dataset) as json_file:
    settings = json.load(json_file)
if not os.path.exists(DIRECTORY):
    os.makedirs(DIRECTORY)


logging.basicConfig(filename=LOG_FILE,
                    filemode="a",
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)

try:
    os.chdir("Ontolearn/examples")
except FileNotFoundError:
    logging.error(FileNotFoundError)
    pass

with open(f'AutoCL/examples/dataset/{DATASET}.json') as json_file:
    settings = json.load(json_file)


def get_data_properties(onto):
    try:
        data_properties = onto.data_properties()
    except Exception as e:
        data_properties = None
        logging.error(e)
    return data_properties


def get_object_properties(onto):
    try:
        object_properties = onto.object_properties()
    except Exception as e:
        object_properties = None
        logging.error(e)
    return object_properties

def create_new_kb_without_features(features):
    prop_object = list(get_object_properties(onto)) + list(get_data_properties(onto))

    for prop in prop_object:
        if prop.name not in features:
            print('prop.name', prop.name)
            destroy_entity(prop)
    onto.save("kb_with_selected_features.owl")
    return KnowledgeBase(path="./kb_with_selected_features.owl")


def get_prominent_properties_occurring_in_top_k_hypothesis(lp, knowledgebase, target_concepts, n=10):
    model = EvoLearner(knowledge_base=knowledgebase)
    model.fit(lp)
    model.save_best_hypothesis(n=n, path=f'Predictions_{target_concepts}')
    hypotheses = list(model.best_hypotheses(n=10))
    prop_object = list(get_object_properties(onto)) + list(get_data_properties(onto))
    print('Prop_Object', prop_object)

    dlr = DLSyntaxObjectRenderer()
    concept_sorted = [dlr.render(c.concept) for c in hypotheses]

    properties = []
    with open(f'{DIRECTORY}{DATASET}_featureSelection.txt', 'a') as f:
        print('Top 10 hypotheses:', file=f)
        for concepts in concept_sorted:
            print("Concepts", concepts, file=f)
            data_object_properties_in_concepts = [prop.name for prop in prop_object if prop.name in concepts]
            properties.extend(data_object_properties_in_concepts)
    properties_from_concepts = list(set(properties))
    print(f'Properties Selected:{properties_from_concepts}')
    return properties_from_concepts


if __name__ == "__main__":
    onto = get_ontology(settings['data_path']).load()
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

        st = time.time()
        relevant_prop = get_prominent_properties_occurring_in_top_k_hypothesis(lp, kb, str_target_concept, n=10)
        st1 = time.time()
        kb_with_selected_features = create_new_kb_without_features(relevant_prop)
        
        model = EvoLearner(knowledge_base=kb_with_selected_features)
        model.fit(lp)
        model.save_best_hypothesis(n=3, path=f'Predictions_{str_target_concept}')
        hypotheses = list(model.best_hypotheses(n=1))
        predictions = model.predict(individuals=list(test_pos | test_neg),
                                    hypotheses=hypotheses)
        f1_score, accuracy = calc_prediction(predictions, test_pos, test_neg)
        quality = hypotheses[0].quality
        
        et = time.time()
        elapsed_time = et - st
        elapsed_time_2 = et - st1
        
        with open(f'{DIRECTORY}{DATASET}_featureSelection.txt', 'a') as f:
            print('str_target_concept', str_target_concept,file=f)
            print('F1 Score', f1_score[1], file=f)
            print('Accuracy', accuracy[1], file=f)
            print('selected feature', relevant_prop, file=f)
            print('Time Taken', elapsed_time, file=f)
            print('Time Taken 2', elapsed_time_2, file=f)
            print('_________________________________________________',file=f)