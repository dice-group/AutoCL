
from ontolearn.concept_learner import CELOE
from typing import Optional
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.metrics import Accuracy, F1
from ontolearn.abstracts import AbstractScorer, BaseRefinement, AbstractHeuristic
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.search import OENode

class celoeWrapper:
    """
    kb: KnowledgeBase
    max_he: int
    min_he: int
    best_only: bool
    calculate_min_max: bool

    search_tree: Dict[OWLClassExpression, TreeNode[OENode]]
    seen_norm_concepts: Set[OWLClassExpression]
    heuristic_queue: 'SortedSet[OENode]'
    best_descriptions: EvaluatedDescriptionSet[OENode, QualityOrderedNode]
    _learning_problem: Optional[EncodedPosNegLPStandardKind]
   """
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
           
        self.celoe=CELOE
        self.knowledge_base=knowledge_base
        self.refinement_operator=refinement_operator
        self.quality_func=self.transform_quality_func(quality_func)        
        self.heuristic_func=heuristic_func
        self.terminate_on_goal= terminate_on_goal
        self.iter_bound=iter_bound       
        self.max_num_of_concepts_tested=max_num_of_concepts_tested#
        self.max_runtime=max_runtime
        self.max_results=max_results
        self.best_only=best_only
        self.calculate_min_max=calculate_min_max


    def transform_quality_func(self, quality_func):
        if quality_func == 'F1':
            quality_func = F1()
        else:
            quality_func = Accuracy()
        return quality_func     

  
    def get_celoe_model(self):
        model = self.celoe(self.knowledge_base,
        self.refinement_operator,
        self.quality_func,         
        self.heuristic_func,
        self.terminate_on_goal,
        self.iter_bound,           
        self.max_num_of_concepts_tested,
        self.max_runtime,
        self.max_results,
        self.best_only,
        self.calculate_min_max )
        return model
