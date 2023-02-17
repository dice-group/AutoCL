from ray import tune
import ray
import json
import os
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.concept_learner import EvoLearner
from ontolearn.learning_problem import PosNegLPStandard
from owlapy import model
from owlapy.model import OWLClass, OWLNamedIndividual, IRI
from ontolearn.utils import setup_logging
from rdflib import OWL, Graph
from ontolearn.metrics import *
from sklearn.model_selection import train_test_split
from random import shuffle
 

    



def my_func(config):

   model=EvoLearner(knowledge_base=target_kb,max_runtime=600,tournament_size=['a'],card_limit=config['b'],population_size=config['c'],num_generations=config['d'],height_limit=config['e'])
   
   tune.report(val=model)
   """"
   
    model.fit(lp, verbose=False)
    model.save_best_hypothesis(n=1, path='Predictions_{0}'.format(str_target_concept))
    # Get Top n hypotheses
    hypotheses = list(model.best_hypotheses(n=1))
    predictions = model.predict(individuals=list(train_pos | train_neg),
                                hypotheses=hypotheses)
    # Use hypotheses as binary function to label individuals.
    
    [print(_) for _ in hypotheses]
    concepts_sorted = sorted(predictions)   
    concepts_dict = {}    
    for con in concepts_sorted:        
           positive_indivuals = predictions[predictions[con].values > 0.0].index.values
           negative_indivuals = predictions[predictions[con].values <= 0.0].index.values
           concepts_dict[con] = {"Pos":positive_indivuals,"Neg":negative_indivuals}        

    for key in concepts_dict:        
           tp = len(list(set(get_short_names(list(typed_pos))).intersection(set(concepts_dict[key]["Pos"]))))
           tn = len(list(set(get_short_names(list(typed_neg))).intersection(set(concepts_dict[key]["Neg"]))))
           fp = len(list(set(get_short_names(list(typed_neg))).intersection(set(concepts_dict[key]["Pos"]))))
           fn = len(list(set(get_short_names(list(typed_pos))).intersection(set(concepts_dict[key]["Neg"]))))
           f1 = F1()
           accuracy = Accuracy()
           f1_score = list(f1.score2(tp=tp,fn=fn,fp=fp,tn=tn))
           accuracy_score = list(accuracy.score2(tp=tp,fn=fn,fp=fp,tn=tn))
           concept_and_score = [key,f1_score] 
          #print(f1_score[1]) 
           f1_fine=f1_score[1]
           acc_fine=accuracy_score[1]
           """
        
    
    
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
    

    typed_pos = list(set(map(OWLNamedIndividual, map(IRI.create, p))))
    typed_neg = list(set(map(OWLNamedIndividual, map(IRI.create, n))))


  
    #Split the data into Training Set and Test Set
    train_pos = set(typed_pos[:int(len(typed_pos)*0.8)])
    train_neg = set(typed_neg[:int(len(typed_neg)*0.8)])
    test_pos = set(typed_pos[-int(len(typed_pos)*0.2):])
    test_neg = set(typed_neg[-int(len(typed_neg)*0.2):])

    lp = PosNegLPStandard(pos=train_pos, neg=train_neg)

    
config_para = {
    'a': tune.grid_search([1, 100]),
    'b': tune.grid_search([1, 100]),
    'c': tune.grid_search([1, 100]),
    'd': tune.grid_search([1, 100]),
    'e': tune.grid_search([1, 100])
}


# 3. Start a Tune run and print the best result.
result = tune.run(my_func, config=config_para)
print("kb"+kb.all_individuals_set)
###the tuner did  not run

print(result.results_df)
   


