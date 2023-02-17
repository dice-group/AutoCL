
from ontolearn.concept_learner import CELOE
from typing import Optional
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.metrics import Accuracy, F1
from ontolearn.utils import setup_logging
from ontolearn.heuristics import CELOEHeuristic, DLFOILHeuristic,OCELHeuristic
from ontolearn.abstracts import AbstractFitness, AbstractScorer, BaseRefinement, AbstractHeuristic
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.search import OENode,LBLNode
from owlapy.model import OWLClassExpression, OWLDataProperty, OWLLiteral, OWLNamedIndividual

class celoeWrapper:
    def __init__(self,
                 knowledge_base: KnowledgeBase,
                 refinement_operator: Optional[BaseRefinement[OENode]] = None,
                 quality_func: Optional[AbstractScorer] = None,
                 heuristic_func: Optional[AbstractHeuristic] = None,
                 terminate_on_goal: Optional[bool] = None,
                 iter_bound: Optional[int] = None,
                 max_num_of_concepts_tested: Optional[int] = None,
                 max_runtime: Optional[int] = None,
                 max_results: int = 10,
                 best_only: bool = False,
                 calculate_min_max: bool = True):
           
        self.ocel=CELOE
        self.knowledge_base=knowledge_base
        self.refinement_operator=refinement_operator #
        self.quality_func=self.transform_quality_func(quality_func)        
        self.heuristic_func=heuristic_func#
        self.terminate_on_goal= terminate_on_goal,#
        self.iter_bound=iter_bound#            
        self.max_num_of_concepts_tested=max_num_of_concepts_tested,#
        self.max_runtime=max_runtime#
        self.max_results=max_results
        self.best_only=best_only
        self.calculate_min_max=calculate_min_max  
       # self.CELOEHeuristic=CELOEHeuristic()
        #self.OCELHeuristic=OCELHeuristic()
        #self.DLFOILHeuristic=DLFOILHeuristic()


    def transform_quality_func(self, quality_func):
        if quality_func == 'F1':
            quality_func = F1()
        else:
            quality_func = Accuracy()
        return quality_func     


    """
    def transform_heuristic_func(self,heuristic_func):
        if heuristic_func=='CELOEHeuristic':
           heuristic_func= self.CELOEHeuristic
        if heuristic_func=='OCELHeuristic':
           heuristic_func= self.OCELHeuristic
        if heuristic_func=='DLFOILHeuristic':
           heuristic_func= self.DLFOILHeuristic
        return  heuristic_func
    """


  
    def get_ocel_model(self):
        model = self.ocel(self.knowledge_base,
                                self.refinement_operator,
        self.quality_func,         
        self.heuristic_func,#
        self.terminate_on_goal,#
        self.iter_bound,#            
        self.max_num_of_concepts_tested,#
        self.max_runtime,#
        self.max_results,
        self.best_only,
        self.calculate_min_max )
        return model



name = 'celoe_python'



    def __init__(self,
                 knowledge_base: KnowledgeBase,
                 refinement_operator: Optional[BaseRefinement[OENode]] = None,
                 quality_func: Optional[AbstractScorer] = None,
                 heuristic_func: Optional[AbstractHeuristic] = None,
                 terminate_on_goal: Optional[bool] = None,
                 iter_bound: Optional[int] = None,
                 max_num_of_concepts_tested: Optional[int] = None,
                 max_runtime: Optional[int] = None,
                 max_results: int = 10,
                 best_only: bool = False,
                 calculate_min_max: bool = True):
        super().__init__(knowledge_base=knowledge_base,
                         refinement_operator=refinement_operator,
                         quality_func=quality_func,
                         heuristic_func=heuristic_func,
                         terminate_on_goal=terminate_on_goal,
                         iter_bound=iter_bound,
                         max_num_of_concepts_tested=max_num_of_concepts_tested,
                         max_runtime=max_runtime)

        self.search_tree = dict()
        self.heuristic_queue = SortedSet(key=HeuristicOrderedNode)
        self._seen_norm_concepts = set()
        self.best_descriptions = EvaluatedDescriptionSet(max_size=max_results, ordering=QualityOrderedNode)

        self.best_only = best_only
        self.calculate_min_max = calculate_min_max

        self.max_he = 0
        self.min_he = 1
        # TODO: CD: This could be defined in BaseConceptLearner as it is used in all classes that inherits from
        # TODO: CD: BaseConceptLearner
        self._learning_problem = None
        self._max_runtime = None