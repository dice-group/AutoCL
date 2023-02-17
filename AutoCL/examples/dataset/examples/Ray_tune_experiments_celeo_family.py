from ray import tune
import ray
import json
import random
import os
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.concept_learner import CELOE
from ontolearn.core.owl.utils import OWLClassExpressionLengthMetric  # noqa: F401
from ontolearn.heuristics import CELOEHeuristic
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.metrics import Accuracy
from owlapy.model import OWLClass, OWLNamedIndividual, IRI
from ontolearn.refinement_operators import ModifiedCELOERefinement
from ontolearn.utils import setup_logging





globalsetting=" "

def get_short_names(individuals):
    short_names = []
    for individual in individuals :
        sn = individual.get_iri().get_short_form()
        short_names.append(sn)

    return short_names


    
def my_func(config):
    print("here is the lp"+lp)

    model = CELOE(knowledge_base=target_kb,
                  max_runtime=600,
                  refinement_operator=op,
                  quality_func=qual,
                  heuristic_func=heur,
                  iter_bound=config['a'],
                  max_num_of_concepts_tested=10_000_000_000,
                  )
    tune.report(val=model)

    
    # exit(1)





if __name__ == '__main__':

    
 print("begining here")
 ray.init()


   

 ##the prepare of the model begining 
setup_logging()
  
try:
    os.chdir("examples")
except FileNotFoundError:
    pass
#the dataset
with open('synthetic_problems.json') as json_file:
    settings = json.load(json_file)
    #####here can set the training and testing dataset
    

kb = KnowledgeBase(path=settings['data_path'])
 
#globalsetting=settings

    
random.seed(0)

for str_target_concept, examples in settings['problems'].items():
    p = set(examples['positive_examples'])
    n = set(examples['negative_examples'])
    print('Target concept: ', str_target_concept)
    
    if str_target_concept in ['Granddaughter', 'Aunt', 'Sister']:
        NS = 'http://www.benchmark.org/family#'
        concepts_to_ignore = {
            OWLClass(IRI(NS, 'Brother')),
            OWLClass(IRI(NS, 'Sister')),
            OWLClass(IRI(NS, 'Daughter')),
            OWLClass(IRI(NS, 'Mother')),
            OWLClass(IRI(NS, 'Grandmother')),
            OWLClass(IRI(NS, 'Father')),
            OWLClass(IRI(NS, 'Grandparent')),
            OWLClass(IRI(NS, 'PersonWithASibling')),
            OWLClass(IRI(NS, 'Granddaughter')),
            OWLClass(IRI(NS, 'Son')),
            OWLClass(IRI(NS, 'Child')),
            OWLClass(IRI(NS, 'Grandson')),
            OWLClass(IRI(NS, 'Grandfather')),
            OWLClass(IRI(NS, 'Grandchild')),
            OWLClass(IRI(NS, 'Parent')),
        }
        target_kb = kb.ignore_and_copy(ignored_classes=concepts_to_ignore)
    else:
        target_kb = kb 


        print("kb is here"+kb.path)
    

    typed_pos = list(set(map(OWLNamedIndividual, map(IRI.create, p))))
    typed_neg = list(set(map(OWLNamedIndividual, map(IRI.create, n))))


  
    #Split the data into Training Set and Test Set
    train_pos = set(typed_pos[:int(len(typed_pos)*0.8)])
    train_neg = set(typed_neg[:int(len(typed_neg)*0.8)])
    test_pos = set(typed_pos[-int(len(typed_pos)*0.2):])
    test_neg = set(typed_neg[-int(len(typed_neg)*0.2):])

    lp = PosNegLPStandard(pos=train_pos, neg=train_neg)
    qual = Accuracy()
    heur = CELOEHeuristic(expansionPenaltyFactor=0.05, startNodeBonus=1.0, nodeRefinementPenalty=0.01)
    op = ModifiedCELOERefinement(knowledge_base=target_kb, use_negation=False, use_all_constructor=False)

    
config_para = {
    'a': tune.grid_search([1, 100])
}


# 3. Start a Tune run and print the best result.
result = tune.run(my_func, config=config_para)
print("kb"+kb.all_individuals_set)
###the tuner did  not run

print(result.results_df)
   


