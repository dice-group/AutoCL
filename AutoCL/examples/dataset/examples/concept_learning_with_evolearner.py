
import json
import os
import pandas as pd
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.concept_learner import EvoLearner
from ontolearn.learning_problem import PosNegLPStandard
from owlapy.model import OWLClass, OWLNamedIndividual, IRI
from ontolearn.utils import setup_logging
from rdflib import OWL, Graph
from ontolearn.metrics import *
from sklearn.model_selection import train_test_split
from random import shuffle

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
conceptlist= ['Aunt','Brother','Cousin','Granddaughter','Uncle','Grandgrandfather']
owlpathlist=list()
for concept in conceptlist:
    owlpathlist.append("/Users/ljymacbook/Ontolearn/examples/Predictions_"+str(concept)+ '.owl')

def get_short_names(individuals):
    short_names = []
    for individual in individuals :
        sn = individual.get_iri().get_short_form()
        short_names.append(sn)

    return short_names
  
   #tournament_size: int = 7,
                 #card_limit: int = 10,
                 #population_size: int = 800,
                 #num_generations: int = 200,
                 #height_limit: int = 17,
#create one empty csv
##df = pd.DataFrame(columns=['target','tournament_size', 'card_limit', 'population_size','num_generations','height_limit','acc','F1-score'])
##df.to_csv('/Users/ljymacbook/Desktop/evolearner-results.csv',mode='a')
conlist=[]
alist=[]
blist=[]
clist=[]
dlist=[]
elist=[]
acclist=[]
f1list=[]
num=0

# noinspection DuplicatedCode
for str_target_concept, examples in settings['problems'].items():
    p = set(examples['positive_examples'])
    n = set(examples['negative_examples'])
    print('Target concept: ', str_target_concept)

    # lets inject more background info
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


    

   # pos_train,pos_test = train_test_split(list(typed_pos),test_size=0.3)
   # neg_train,neg_test = train_test_split(list(typed_neg),test_size=0.3)
  

    shuffle(typed_pos)   
    shuffle(typed_neg)
  
    #Split the data into Training Set and Test Set
    train_pos = set(typed_pos[:int(len(typed_pos)*0.8)])
    train_neg = set(typed_neg[:int(len(typed_neg)*0.8)])
    test_pos = set(typed_pos[-int(len(typed_pos)*0.2):])
    test_neg = set(typed_neg[-int(len(typed_neg)*0.2):])

    lp = PosNegLPStandard(pos=train_pos, neg=train_neg)
                
                 #tournament_size: int = 7,
                 #card_limit: int = 10,
                 #population_size: int = 800,
                 #num_generations: int = 200,
                 #height_limit: int = 17,
         ####   grid search start
     
    

    #copy from the function
   
    for a in [1,10]:
     for b in [1,10]:
      for c in [1,10]:
       for d in [1,10]:
        for e in [1,10]:
            
         
          model = EvoLearner(knowledge_base=target_kb,max_runtime=600, tournament_size=a,card_limit=b,population_size=c,num_generations=d,height_limit=e)
         #training
          model.fit(lp, verbose=False)

         

    #model = EvoLearner(knowledge_base=target_kb, max_runtime=600, tournament_size=a,card_limit=b,population_size=c,num_generations=d,height_limit=e)
   # model.fit(lp, verbose=False)
  ##F1 and accuracy
          model.save_best_hypothesis(n=1, path='Predictions_{0}'.format(str_target_concept))
    # Get Top n hypotheses
          hypotheses = list(model.best_hypotheses(n=1))
    # Use hypotheses as binary function to label individuals.
          predictions = model.predict(individuals=list(train_pos | train_neg),
                                hypotheses=hypotheses)
                                
    # print(predictions)
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
          #print(accuracy_score[1])
          ###store in the csv
          #print(a,b,c,d,e)
          #print(str_target_concept) 
         
           conlist.insert(num,str_target_concept)
           alist.insert(num,a)
           blist.insert(num,b)
           clist.insert(num,c)
           dlist.insert(num,d)
           elist.insert(num,e)
           acclist.insert(num,acc_fine)
           f1list.insert(num,f1_fine)
           num=num+1

          #open csv file and write it
          #check index

          #dfread= pd.read_csv('/Users/ljymacbook/Desktop/evolearner-results.csv', 'r') 
          #if(dfread.shape[0]==0):
           # print("the length is 0,insert the data from begining")
           #change the csv function by pandas
"""
          new=pd.DataFrame({'target':str_target_concept,
                  'tournament_size':a,
                  'card_limit':b,
                  'population_size':c,
                  'num_generations':d,
                  'height_limit':e,
                  'acc':acc_fine,
                  'F1-score':f1_fine},index=[num]
                 )   
          print(new)
    
          df=df.append(new,ignore_index=True) 
    #print(dataframe)
    """


  
print(len(conlist),len(alist),len(blist),len(clist),len(dlist),len(elist),len(acclist),len(f1list))
dict = {'target': conlist, 'tournament_size': alist, 'card_limit': blist,'population_size':clist , 'num_generations':dlist,
                  'height_limit':elist,
                  'acc':acclist,
                  'F1-score':f1list}
     
df = pd.DataFrame(dict)
#df.style.highlight_max(subset='acc')

df.to_csv('/Users/ljymacbook/Desktop/evolearner-results.csv')
          #dataframe.to_csv('/Users/ljymacbook/Desktop/evolearner-results.csv')
        #row = {'numer':'','target':str_target_concept,'tournament_size':a, 'card_limit':b, 'population_size':c,'num_generations':d,'height_limit':e,'acc':accuracy_score,'F1-score':f1_score}
        
        #dataframe = pd.DataFrame(row)
        #print(dataframe)
            # Open the CSV file in "append" mode
        #with open('/Users/ljymacbook/Desktop/evolearner-results.csv', 'a', newline='') as f:
        # writer = csv.DictWriter(f, fieldnames=row.keys())
        
    # Append single row to CSV
        # writer.writerow(row) 
         #find the best hyperparamters
          
            #dfread.loc['new']=[str_target_concept,a,b,c,d,e,accuracy_score,f1_score]
            #=pd.DataFrame([str_target_concept,a,b,c,d,e,accuracy_score,f1_score]).T
            #else
            #write in the index=length+1 





