import json
import os
import time
from random import shuffle
import logging
from owlready2 import get_ontology, default_world, destroy_entity

from examples.search import calc_prediction
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.owlapy.model import OWLNamedIndividual, IRI
from ontolearn.concept_learner import EvoLearner
from ontolearn.owlapy.render import DLSyntaxObjectRenderer
from wrapper_celoe import CeloeWrapper

DIRECTORY = './ICML/FINAL/FS_WITH_EVO_ON_CELOE/'
LOG_FILE = 'featureSelectionWithGraph.log'
DATASET = 'mammograph'

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


PATH = f'dataset/{DATASET}.json'
with open(PATH) as json_file:
    settings = json.load(json_file)
if not os.path.exists(DIRECTORY):
    os.makedirs(DIRECTORY)


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
            # print('prop.name', prop.name)
            destroy_entity(prop)
    onto.save("kb_with_selected_features.owl")
    return KnowledgeBase(path="./kb_with_selected_features.owl")


def get_prominent_properties_occurring_in_top_k_hypothesis(lp, knowledgebase, target_concepts, n=10):
    min_properties_count = 5
    max_hypothesis_count = 800
    current_hypothesis_count = n
    properties_from_concepts = []
    model = EvoLearner(knowledge_base=knowledgebase)
    model.fit(lp)
    count = 1
    counter = 0
    previous_properties_list = []
    prop_object = list(get_object_properties(onto)) + list(get_data_properties(onto))

    while (len(properties_from_concepts) < min_properties_count and
           current_hypothesis_count <= max_hypothesis_count and
           counter < 100):

        with open(f'{DIRECTORY}{DATASET}_featureSelection.txt', 'a') as f:
            model.save_best_hypothesis(n=current_hypothesis_count, path=f'Predictions_{target_concepts}')
            hypotheses = list(model.best_hypotheses(n=current_hypothesis_count))

            dlr = DLSyntaxObjectRenderer()
            concept_sorted = [dlr.render(c.concept) for c in hypotheses]

            properties = []
            for concepts in concept_sorted:
                data_object_properties_in_concepts = [prop.name for prop in prop_object if prop.name in concepts]
                properties.extend(data_object_properties_in_concepts)

            properties_from_concepts = list(set(properties))

            if properties_from_concepts == previous_properties_list:
                counter += 1
            else:
                counter = 0

            previous_properties_list = properties_from_concepts.copy()

            current_hypothesis_count += 1

    if len(properties_from_concepts) >= min_properties_count:
        with open(f'{DIRECTORY}{DATASET}_featureSelection.txt', 'a') as f:
            for concepts in concept_sorted:
                print('Length of Property:', len(properties_from_concepts), file=f)
                print("Concepts", concepts, file=f)
                print('Count', count, file=f)
            print('Selected Props', properties_from_concepts, file=f)

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
        et = time.time()
        elapsed_time = et - st

        new_kb = create_new_kb_without_features(relevant_prop)

        print('Process Started')
        wrap_obj = CeloeWrapper(knowledge_base=new_kb,
                                max_runtime=600,
                                max_num_of_concepts_tested=10_000_000_000,
                                iter_bound=10_000_000_000)
        st1 = time.time()
        wrap_obj = CeloeWrapper(
            knowledge_base=new_kb,
            max_runtime=600,
            max_num_of_concepts_tested=10_000_000,
            iter_bound=10_000_000
        )
        model = wrap_obj.get_celoe_model()
        model.fit(lp, verbose=False)
        model.save_best_hypothesis(n=3, path=f'Predictions_{str_target_concept}')
        hypotheses = list(model.best_hypotheses(n=1))
        predictions = model.predict(individuals=list(test_pos | test_neg), hypotheses=hypotheses)
        f1_score, accuracy = calc_prediction(predictions, test_pos, test_neg)
        quality = hypotheses[0].quality

        et1 = time.time()
        elapsed_time_2 = et1 - st1

        with open(f'{DIRECTORY}{DATASET}_featureSelection.txt', 'a') as f:
            print('str_target_concept', str_target_concept, file=f)
            print('F1 Score:', f1_score[1], file=f)
            print('Accuracy:', accuracy[1], file=f)
            print('selected feature:', relevant_prop, file=f)
            print('Time Taken for FS:', elapsed_time, file=f)
            print('Time Taken for fit and predict:', elapsed_time_2, file=f)
