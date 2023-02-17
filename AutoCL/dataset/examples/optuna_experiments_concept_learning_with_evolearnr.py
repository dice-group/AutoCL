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
import optuna
import pandas as pd

 

def get_short_names(individuals):
    short_names = []
    for individual in individuals :
        sn = individual.get_iri().get_short_form()
        short_names.append(sn)

    return short_names



def my_func(trial):

    param_grid = {
        "a": trial.suggest_int("a", 1, 100),
        "b": trial.suggest_int("b", 1, 100),
        "c": trial.suggest_int("c", 1, 100),
        "d": trial.suggest_int("d", 1, 100),
        "e": trial.suggest_int("e", 1,100)
      }


      
    num=0
    model=EvoLearner(knowledge_base=target_kb,max_runtime=600,tournament_size=param_grid['a'],card_limit=param_grid['b'],population_size=param_grid['c'],num_generations=param_grid['d'],height_limit=param_grid['e'])
   
  
    
   
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

           conlist.insert(num,str_target_concept)
           alist.insert(num,param_grid['a'])
           blist.insert(num,param_grid['b'])
           clist.insert(num,param_grid['c'])
           dlist.insert(num,param_grid['d'])
           elist.insert(num,param_grid['e'])
           acclist.insert(num,acc_fine)
           f1list.insert(num,f1_fine)
           num=num+1

    print(len(conlist),len(alist),len(blist),len(clist),len(dlist),len(elist),len(acclist),len(f1list))
    dict = {'target': conlist, 'tournament_size': alist, 'card_limit': blist,'population_size':clist , 'num_generations':dlist,
                  'height_limit':elist,
                  'acc':acclist,
                  'F1-score':f1list}
     
    df = pd.DataFrame(dict)
#df.style.highlight_max(subset='acc')

    df.to_csv('/Users/ljymacbook/Desktop/evolearner-results_mammo_optuna_gridsearch.csv')  
    
    quality = hypotheses[0].quality
    return quality
   
        
    
    
if __name__ == '__main__':





    
 print("begining here")  
 conlist=[]
 alist=[]
 blist=[]
 clist=[]
 dlist=[]
 elist=[]
 acclist=[]
 f1list=[]
 

 ##the prepare of the model begining 
setup_logging()
  
try:
    os.chdir("examples")
except FileNotFoundError:
    pass
#the dataset
with open('synthetic_problems_mammographic.json') as json_file:
    settings = json.load(json_file)
    #####here can set the training and testing dataset
    

kb = KnowledgeBase(path=settings['data_path'])
 
#globalsetting=settings

    


for str_target_concept, examples in settings['problems'].items():
    p = set(examples['positive_examples'])
    n = set(examples['negative_examples'])
    print('Target concept: ', str_target_concept)
    
 
    target_kb = kb 
    

    typed_pos = list(set(map(OWLNamedIndividual, map(IRI.create, p))))
    typed_neg = list(set(map(OWLNamedIndividual, map(IRI.create, n))))


  
    #Split the data into Training Set and Test Set
    train_pos = set(typed_pos[:int(len(typed_pos)*0.8)])
    train_neg = set(typed_neg[:int(len(typed_neg)*0.8)])
    test_pos = set(typed_pos[-int(len(typed_pos)*0.2):])
    test_neg = set(typed_neg[-int(len(typed_neg)*0.2):])

    lp = PosNegLPStandard(pos=train_pos, neg=train_neg)
    
    search_space = {"a": [1,10,100,1000],
                "b": [1,10,100,1000],
                "c": [1,10,100,1000],
                "d": [1,10,100,1000],
                "e": [1,10,100,1000],
            }


#study = optuna.create_study(direction="maximize", study_name="evolearner")
#study = optuna.create_study(direction="maximize",sampler=optuna.samplers.RandomSampler())
study = optuna.create_study(direction="maximize",sampler=optuna.samplers.GridSampler(search_space))
#only study one concept problem,eg,only grandgrandfather
study.optimize(my_func,n_trials=100)
   


