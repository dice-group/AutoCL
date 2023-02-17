
from rdflib import Graph

g = Graph()
g.parse("/Users/ljymacbook/Ontolearn/examples/Predictions_Aunt.owl", format="xml")
#with open("/Users/ljymacbook/Ontolearn/examples/Predictions_AuntONE.csv","w",newline='') as f1:
header=['Subject','Predicate','Object']

for (s,p,o) in g:
       # print(stmt)
        list_triple=[]
        if str(s)[-6:][:4] == 'Pred' and str(p)[-8:][:2] == 'f1':
         print(o)

             

 
  




