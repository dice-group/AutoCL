import os

from ontolearn.concept_learner import CELOE, OCEL
from ontolearn.metrics import Accuracy, F1
from ontolearn.abstracts import AbstractScorer, BaseRefinement, AbstractHeuristic, AbstractFitness
from ontolearn.search import OENode
from ontolearn.concept_learner import EvoLearner
from typing import Optional
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.value_splitter import BinningValueSplitter, EntropyValueSplitter
from ontolearn.ea_initialization import AbstractEAInitialization
from ontolearn.ea_algorithms import AbstractEvolutionaryAlgorithm


class CeloeWrapper:
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

        self.celoe = CELOE
        self.knowledge_base = knowledge_base
        self.refinement_operator = refinement_operator
        self.quality_func = self.transform_quality_func(quality_func)
        self.heuristic_func = heuristic_func
        self.terminate_on_goal = terminate_on_goal
        self.iter_bound = iter_bound
        self.max_num_of_concepts_tested = max_num_of_concepts_tested  #
        self.max_runtime = max_runtime
        self.max_results = max_results
        self.best_only = best_only
        self.calculate_min_max = calculate_min_max

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
                           self.calculate_min_max
                           )
        return model

    @classmethod
    def reinitialize_with_best_hpo(cls, best_hpo, new_kb):
        if not best_hpo.empty:
            return cls(
                knowledge_base=new_kb,
                max_runtime=int(best_hpo['max_runtime'].values[0]),
                max_num_of_concepts_tested=int(best_hpo['max_num_of_concepts_tested'].values[0]),
                iter_bound=int(best_hpo['iter_bound'].values[0]),
                quality_func=str(best_hpo['quality_func'].values[0])
            )

    def execute(self, str_target_concept, lp, test_pos, test_neg):

        model = self.get_celoe_model()
        model.fit(lp)
        # Get the directory of the current script
        current_script_dir = os.path.dirname(os.path.abspath(__file__))

        # Get the parent directory of the current script's directory
        parent_dir = os.path.abspath(os.path.join(current_script_dir, os.pardir))

        # Define the path to the results folder inside the parent directory
        results_folder = os.path.join(parent_dir, "results")

        # Ensure the results folder exists
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)

        model.save_best_hypothesis(n=3, path=results_folder + '/Prediction_' + str_target_concept)

        # Get the best hypothesis
        hypotheses = list(model.best_hypotheses(n=1))

        # Make predictions on the test data
        predictions = model.predict(individuals=list(test_pos | test_neg), hypotheses=hypotheses)

        return predictions, hypotheses


class EvoLearnerWrapper:
    def __init__(self, knowledge_base: KnowledgeBase,
                 quality_func: Optional[AbstractScorer] = None,
                 fitness_func: Optional[AbstractFitness] = None,
                 init_method: Optional[AbstractEAInitialization] = None,
                 algorithm: Optional[AbstractEvolutionaryAlgorithm] = None,
                 mut_uniform_gen: Optional[AbstractEAInitialization] = None,
                 value_splitter: Optional[str] = None,
                 terminate_on_goal: Optional[bool] = None,
                 max_runtime: Optional[int] = None,
                 use_data_properties: Optional[str] = 'True',
                 use_card_restrictions: bool = True,
                 use_inverse: Optional[str] = 'False',
                 tournament_size: int = 7,
                 card_limit: int = 10,
                 population_size: int = 800,
                 num_generations: int = 200,
                 height_limit: int = 17):
        self.evolearner = EvoLearner
        self.binning_value_splitter = BinningValueSplitter()
        self.entropy_value_splitter = EntropyValueSplitter()
        self.knowledge_base = knowledge_base
        self.quality_func = self.transform_quality_func(quality_func)
        self.fitness_func = fitness_func
        self.init_method = init_method
        self.terminate_on_goal = terminate_on_goal
        self.algorithm = algorithm
        self.mut_uniform_gen = mut_uniform_gen
        self.value_splitter = self.transform_value_splitter(value_splitter)
        self.use_data_properties = self.transform_use_data_properties(use_data_properties)
        self.use_card_restrictions = use_card_restrictions
        self.max_runtime = max_runtime
        self.use_inverse = self.transform_use_inverse(use_inverse)
        self.tournament_size = tournament_size
        self.card_limit = card_limit
        self.population_size = population_size
        self.num_generations = num_generations
        self.height_limit = height_limit

    def transform_value_splitter(self, value_splitter):
        if value_splitter == 'entropy_value_splitter':
            value_splitter = self.entropy_value_splitter
        elif value_splitter == 'binning_value_splitter':
            value_splitter = self.binning_value_splitter
        else:
            value_splitter = None
        return value_splitter

    def transform_use_data_properties(self, use_data_properties):
        if use_data_properties == 'False':
            use_data_properties = False
        else:
            use_data_properties = True
        return use_data_properties

    def transform_use_inverse(self, use_inverse):
        if use_inverse == 'True':
            use_inverse = True
        else:
            use_inverse = False
        return use_inverse

    def transform_quality_func(self, quality_func):
        if quality_func == 'F1':
            quality_func = F1()
        else:
            quality_func = Accuracy()
        return quality_func

    def get_evolearner_model(self):
        model = self.evolearner(self.knowledge_base,
                                self.quality_func,
                                self.fitness_func,
                                self.init_method,
                                self.algorithm,
                                self.mut_uniform_gen,
                                self.value_splitter,
                                self.terminate_on_goal,
                                self.max_runtime,
                                self.use_data_properties,
                                self.use_card_restrictions,
                                self.use_inverse,
                                self.tournament_size,
                                self.card_limit,
                                self.population_size,
                                self.num_generations,
                                self.height_limit)
        return model

    @classmethod
    def reinitialize_with_best_hpo(cls, best_hpo, new_kb):
        if not best_hpo.empty:
            return cls(
                knowledge_base=new_kb,
                max_runtime=int(best_hpo['max_runtime'].values[0]),
                tournament_size=int(best_hpo['tournament_size'].values[0]),
                height_limit=int(best_hpo['height_limit'].values[0]),
                card_limit=int(best_hpo['card_limit'].values[0]),
                use_data_properties=str(best_hpo['use_data_properties'].values[0]),
                use_inverse=str(best_hpo['use_inverse_prop'].values[0]),
                quality_func=str(best_hpo['quality_func'].values[0]),
                value_splitter=str(best_hpo['value_splitter'].values[0])
            )

    def execute(self, str_target_concept, lp, test_pos, test_neg):
        model = self.get_evolearner_model()
        model.fit(lp)
        # Get the directory of the current script
        current_script_dir = os.path.dirname(os.path.abspath(__file__))

        # Get the parent directory of the current script's directory
        parent_dir = os.path.abspath(os.path.join(current_script_dir, os.pardir))

        # Define the path to the results folder inside the parent directory
        results_folder = os.path.join(parent_dir, "results")

        # Ensure the results folder exists
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)

        model.save_best_hypothesis(n=3, path=results_folder + '/Prediction_' + str_target_concept)

        # Get the best hypothesis
        hypotheses = list(model.best_hypotheses(n=1))

        # Make predictions on the test data
        predictions = model.predict(individuals=list(test_pos | test_neg), hypotheses=hypotheses)

        return predictions, hypotheses


class OcelWrapper:
    def __init__(self,
                 knowledge_base: KnowledgeBase,
                 refinement_operator: Optional[BaseRefinement[OENode]] = None,
                 quality_func: Optional[AbstractScorer] = None,
                 heuristic_func: Optional[AbstractHeuristic] = None,
                 terminate_on_goal: Optional[bool] = None,
                 iter_bound: Optional[int] = None,
                 max_runtime: Optional[int] = None,
                 max_num_of_concepts_tested: Optional[int] = None,
                 max_results: int = 10,
                 best_only: bool = False,
                 calculate_min_max: bool = True):

        self.ocel = OCEL
        self.knowledge_base = knowledge_base
        self.refinement_operator = refinement_operator
        self.quality_func = self.transform_quality_func(quality_func)
        self.heuristic_func = heuristic_func
        self.terminate_on_goal = terminate_on_goal
        self.iter_bound = iter_bound
        self.max_num_of_concepts_tested = max_num_of_concepts_tested
        self.max_runtime = max_runtime
        self.max_results = max_results
        self.best_only = best_only
        self.calculate_min_max = calculate_min_max

    def transform_quality_func(self, quality_func):
        if quality_func == 'F1':
            quality_func = F1()
        else:
            quality_func = Accuracy()
        return quality_func

    def get_ocel_model(self):
        model = self.ocel(self.knowledge_base,
                          self.refinement_operator,
                          self.quality_func,
                          self.heuristic_func,
                          self.terminate_on_goal,
                          self.iter_bound,
                          self.max_num_of_concepts_tested,
                          self.max_runtime,
                          self.max_results,
                          self.best_only,
                          self.calculate_min_max
                          )
        return model

    @classmethod
    def reinitialize_with_best_hpo(cls, best_hpo, new_kb):
        if not best_hpo.empty:
            return cls(
                knowledge_base=new_kb,
                max_runtime=int(best_hpo['max_runtime'].values[0]),
                max_num_of_concepts_tested=int(best_hpo['max_num_of_concepts_tested'].values[0]),
                iter_bound=int(best_hpo['iter_bound'].values[0]),
                quality_func=str(best_hpo['quality_func'].values[0])
            )

    def execute(self, str_target_concept, lp, test_pos, test_neg):
        model = self.get_ocel_model()
        model.fit(lp)
        # Get the directory of the current script
        current_script_dir = os.path.dirname(os.path.abspath(__file__))

        # Get the parent directory of the current script's directory
        parent_dir = os.path.abspath(os.path.join(current_script_dir, os.pardir))

        # Define the path to the results folder inside the parent directory
        results_folder = os.path.join(parent_dir, "results")

        # Ensure the results folder exists
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)

        model.save_best_hypothesis(n=3, path=results_folder + '/Prediction_' + str_target_concept)

        # Get the best hypothesis
        hypotheses = list(model.best_hypotheses(n=1))

        # Make predictions on the test data
        predictions = model.predict(individuals=list(test_pos | test_neg), hypotheses=hypotheses)

        return predictions, hypotheses
