import json
import os
import time
import logging
import optuna
import pandas as pd
import numpy as np
from random import shuffle

from optuna.samplers import RandomSampler
from optuna.samplers import TPESampler
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.owlapy.model import OWLNamedIndividual, IRI, OWLReasoner
from wrapper_celoe import CeloeWrapper
from search import calc_prediction


LOG_FILE = "hpo_celoe.log"
DATASET = "mutagenesis"
logging.basicConfig(filename=LOG_FILE,
                    filemode="a",
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)

try:
    os.chdir("Ontolearn/examples")
except FileNotFoundError:
    pass

PATH = f'dataset/{DATASET}.json'
with open(PATH) as json_file:
    settings = json.load(json_file)

kb = KnowledgeBase(path=settings['data_path'])
df = pd.DataFrame(columns=['LP', 'max_runtime', 
                           'max_num_of_concepts_tested', 'iter_bound', 'quality_func',
                           'quality_score', 'Validation_f1_Score',
                           'Validation_accuracy'])


class OptunaSamplers():
    def __init__(self, dataset, kb, lp, concept, val_pos, val_neg, df):
        # For grid sampler specify the search space
        search_space = {'max_runtime': [1, 10, 50, 100],
                        'max_num_of_concepts_tested': [2, 5, 15],
                        'iter_bound': [3, 10, 25]
                        }
        self.lp = lp
        self.concept = concept
        self.val_pos = val_pos
        self.val_neg = val_neg
        self.dataset = dataset
        self.kb = kb
        self.df = df
        self.study_random_sampler = optuna.create_study(sampler=RandomSampler(),
                                                        direction='maximize')
        self.study_grid_sampler = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space=search_space))
        self.study_tpe_sampler = optuna.create_study(sampler=TPESampler(),
                                                     direction='maximize')
        self.sampler = optuna.samplers.CmaEsSampler()
        self.nsgii = optuna.samplers.NSGAIISampler()
        self.qmc_sampler = optuna.samplers.QMCSampler()
        self.study_cmaes_sampler = optuna.create_study(sampler=self.sampler)
        self.study_nsgaii_sampler = optuna.create_study(sampler=self.nsgii)
        self.study_qmc_sampler = optuna.create_study(sampler=self.qmc_sampler)

    def write_to_df(self, **space):
        self.df.loc[len(self.df.index)] = [self.concept, space['max_runtime'],
                                           space['max_num_of_concepts_tested'], space['iter_bound'],
                                           space['quality_func'],
                                           space['quality_score'],
                                           space['Validation_f1_Score'], space['Validation_accuracy']]

    def convert_to_csv(self):
        timestr = str(time.strftime("%Y%m%d-%H%M%S"))
        filename = f'{self.dataset}_CELOE_Output {timestr}'
        self.df.to_csv(filename+".csv", index=False)

    def objective(self, trial):
        max_runtime = trial.suggest_int("max_runtime", 2, 600)
        max_num_of_concepts_tested = trial.suggest_int('max_num_of_concepts_tested', 2, 10_000_000_00)
        iter_bound = trial.suggest_int('iter_bound', 3, 10_000_000_00)

        # call the wrapper class
        wrap_obj = CeloeWrapper(knowledge_base=self.kb,
                                max_runtime=max_runtime,
                                max_num_of_concepts_tested=max_num_of_concepts_tested,
                                iter_bound=iter_bound
                                )
        model = wrap_obj.get_celoe_model()
        model.fit(self.lp, verbose=False)
        model.save_best_hypothesis(n=3, path='Predictions_{0}'.format(self.concept))
        hypotheses = list(model.best_hypotheses(n=1))
        predictions = model.predict(individuals=list(self.val_pos | self.val_neg),
                                    hypotheses=hypotheses)
        f1_score, accuracy = calc_prediction(predictions, self.val_pos, self.val_neg)
        quality = hypotheses[0].quality
        return quality

    def objective_with_categorical_distribution(self, trial):
        # categorical distribution efficiently works for the following samplers
        # Random Sampler, Grid Sampler, TPE Sampler and NSGAII Sampler
        max_runtime = trial.suggest_int("max_runtime", 1, 600)
        max_num_of_concepts_tested = trial.suggest_int("max_num_of_concepts_tested", 2, 10_000_000_00)
        iter_bound = trial.suggest_int('iter_bound', 2, 10_000_000_00)
        quality_func = trial.suggest_categorical('quality_func', ['F1', 'Accuracy'])
        
        # call the wrapper class
        wrap_obj = CeloeWrapper(knowledge_base=self.kb,
                                max_runtime=max_runtime,
                                max_num_of_concepts_tested=max_num_of_concepts_tested,
                                iter_bound=iter_bound,
                                quality_func=quality_func,
                                )
        model = wrap_obj.get_celoe_model()
        model.fit(self.lp, verbose=False)
        model.save_best_hypothesis(n=3, path='Predictions_{0}'.format(self.concept))
        hypotheses = list(model.best_hypotheses(n=1))
        predictions = model.predict(individuals=list(self.val_pos | self.val_neg),
                                    hypotheses=hypotheses)
        f1_score, accuracy = calc_prediction(predictions, self.val_pos, self.val_neg)
        print(f'F1_SCORE{f1_score}, Accuracy{accuracy}')
        quality = hypotheses[0].quality

        # create a dictionary
        space = dict()
        space['max_runtime'] = max_runtime
        space['max_num_of_concepts_tested'] = max_num_of_concepts_tested
        space['iter_bound'] = iter_bound
        space['quality_func'] = quality_func
        space['quality_score'] = quality
        space['Validation_f1_Score'] = f1_score[1]
        space['Validation_accuracy'] = accuracy[1]
        self.write_to_df(**space)
        return quality

    def objective_without_categorical_distribution(self, trial):
        # Categorical distribution does not work for QMC and CMEs sampler
        # The categorical distribution values are sampled in such cases
        # using Random Sampler
        # Using a threshold value to categorise the hyperparameters.
        max_runtime = trial.suggest_int("max_runtime", 2, 500)
        max_num_of_concepts_tested = trial.suggest_int("max_num_of_concepts_tested", 2, 10_000_000_000)
        iter_bound = trial.suggest_int('iter_bound', 3, 10_000_000_000)
        
        quality_func = trial.suggest_int('quality_func', 1, 2)   
        quality_func = 'F1' if quality_func >= 2 else 'Accuracy'

        # call the wrapper class
        wrap_obj = CeloeWrapper(knowledge_base=self.kb,
                                max_runtime=max_runtime,
                                max_num_of_concepts_tested=max_num_of_concepts_tested,
                                iter_bound=iter_bound,
                                quality_func=quality_func,
                                )
        model = wrap_obj.get_celoe_model()
        model.fit(self.lp, verbose=False)
        model.save_best_hypothesis(n=3, path='Predictions_{0}'.format(self.concept))
        hypotheses = list(model.best_hypotheses(n=1))
        predictions = model.predict(individuals=list(self.val_pos | self.val_neg),
                                    hypotheses=hypotheses)
        f1_score, accuracy = calc_prediction(predictions, self.val_pos, self.val_neg)
        quality = hypotheses[0].quality

        # create a dictionary
        space = dict()
        space['max_runtime'] = max_runtime
        space['max_num_of_concepts_tested'] = max_num_of_concepts_tested
        space['iter_bound'] = iter_bound
        space['quality_func'] = quality_func
        space['quality_score'] = quality
        space['Validation_f1_Score'] = f1_score[1]
        space['Validation_accuracy'] = accuracy[1]
        self.write_to_df(**space)
        return quality

    def get_best_optimisation_result_for_random_sampler(self, n_trials):
        self.study_random_sampler.optimize(self.objective_with_categorical_distribution, n_trials=n_trials)
        logging.info(f"BEST TRIAL RANDOM SAMPLER : {self.study_random_sampler.best_trial}")
        # best parameter combination
        logging.info(f"BEST PARAMS RANDOM SAMPLER : {self.study_random_sampler.best_params}")
        # score achieved with best parameter combination
        logging.info(f"BEST VALUE RANDOM SAMPLER : {self.study_random_sampler.best_value}")

    def get_best_optimization_result_for_grid_sampler(self, n_trials):
        self.study_grid_sampler.optimize(self.objecfive, n_trials=n_trials)
        logging.info(f"BEST TRIAL GRID SAMPLER : {self.study_grid_sampler.best_trial}")
        # best parameter combination
        logging.info(f"BEST PARAMS GRID SAMPLER : {self.study_grid_sampler.best_params}")
        # score achieved with best parameter combination
        logging.info(f"BEST VALUE GRID SAMPLER : {self.study_grid_sampler.best_value}")

    def get_best_optimization_result_for_tpe_sampler(self, n_trials):
        self.study_tpe_sampler.optimize(self.objective_with_categorical_distribution, n_trials=n_trials)
        logging.info(f"BEST TRIAL TPE SAMPLER : {self.study_tpe_sampler.best_trial}")
        # best parameter combination
        logging.info(f"BEST PARAMS TPE SAMPLER : {self.study_tpe_sampler.best_params}")
        # score achieved with best parameter combination
        logging.info(f"BEST VALUE TPE SAMPLER : {self.study_tpe_sampler.best_value}")

    def get_best_optimization_result_for_cmes_sampler(self, n_trials):
        self.study_cmaes_sampler.optimize(self.objective_without_categorical_distribution, n_trials=n_trials)
        logging.info(f"BEST TRIAL CMAES SAMPLER : {self.study_cmaes_sampler.best_trial}")
        # best parameter combination
        logging.info(f"BEST PARAMS CMAES SAMPLER : {self.study_cmaes_sampler.best_params}")
        # score achieved with best parameter combination
        logging.info(f"BEST VALUE CMAES SAMPLER : {self.study_cmaes_sampler.best_value}")

    def get_best_optimization_result_for_nsgii_sampler(self, population_size):
        self.study_nsgaii_sampler.optimize(self.objective_with_categorical_distribution, n_trials=population_size)
        logging.info(f"BEST TRIAL NSGAII SAMPLER : {self.study_nsgaii_sampler.best_trial}")
        # best parameter combination
        logging.info(f"BEST PARAMS NSGAII SAMPLER : {self.study_nsgaii_sampler.best_params}")
        # score achieved with best parameter combination
        logging.info(f"BEST VALUE NSGAII SAMPLER : {self.study_nsgaii_sampler.best_value}")

    def get_best_optimization_result_for_qmc_sampler(self, n_trials):
        self.study_qmc_sampler.optimize(self.objective_without_categorical_distribution, n_trials=n_trials)
        logging.info(f"BEST TRIAL QMC SAMPLER : {self.study_qmc_sampler.best_trial}")
        # best parameter combination
        logging.info(f"BEST PARAMS QMC SAMPLER : {self.study_qmc_sampler.best_params}")
        # score achieved with best parameter combination
        logging.info(f"BEST VALUE QMC SAMPLER : {self.study_qmc_sampler.best_value}")


if __name__ == "__main__":
    df = pd.DataFrame(columns=['LP', 'max_runtime',
                               'max_num_of_concepts_tested', 'iter_bound', 'quality_func',
                               'quality_score', 'Validation_f1_Score',
                               'Validation_accuracy'])
    for str_target_concept, examples in settings['problems'].items():
        p = set(examples['positive_examples'])
        n = set(examples['negative_examples'])

        typed_pos = list(set(map(OWLNamedIndividual, map(IRI.create, p))))
        typed_neg = list(set(map(OWLNamedIndividual, map(IRI.create, n))))
        
        shuffle(typed_pos)
        shuffle(typed_neg)

        # Split the data into Training Set, Validation Set and Test Set
        train_pos, val_pos, test_pos = np.split(typed_pos,
                                                [int(len(typed_pos)*0.6),
                                                 int(len(typed_pos)*0.8)])
        train_neg, val_neg, test_neg = np.split(typed_neg,
                                                [int(len(typed_neg)*0.6),
                                                 int(len(typed_neg)*0.8)])
        train_pos, train_neg = set(train_pos), set(train_neg)
        val_pos, val_neg = set(val_pos), set(val_neg)
        test_pos, test_neg = set(test_pos), set(test_neg)

        lp = PosNegLPStandard(pos=train_pos, neg=train_neg)

        # create class object and get the optimised result
        optuna1 = OptunaSamplers(lp, str_target_concept, val_pos, val_neg)
        optuna1.get_best_optimization_result_for_tpe_sampler(5)
        optuna1.convert_to_csv(df)

        # get the best hpo
        best_hpo = df.loc[df['Validation_f1_Score'] == df['Validation_f1_Score'].values.max()]
        if len(best_hpo.index) > 1:
            best_hpo = best_hpo.loc[(best_hpo['Validation_accuracy'] == best_hpo['Validation_accuracy'].values.max()) &
                                    (best_hpo['max_runtime'] == best_hpo['max_runtime'].values.min())]
        logging.info(f"BEST HPO : {best_hpo}")

        wrap_obj = CeloeWrapper(knowledge_base=kb,
                                max_runtime=int(best_hpo['max_runtime'].values[0]),
                                max_num_of_concepts_tested=int(best_hpo['max_num_of_concepts_tested'].values[0]),
                                iter_bound=int(best_hpo['iter_bound'].values[0]),
                                quality_func=str(best_hpo['quality_func'].values[0])
                                )
      
        model = wrap_obj.get_celoe_model()
        model.fit(lp, verbose=False)
        model.save_best_hypothesis(n=3, path='Predictions_{0}'.
                                   format(str_target_concept))
        hypotheses = list(model.best_hypotheses(n=1))
        predictions = model.predict(individuals=list(test_pos | test_neg),
                                    hypotheses=hypotheses)
        f1_score, accuracy = calc_prediction(predictions, test_pos, test_neg)
        quality = hypotheses[0].quality

        with open(f'{DATASET}.txt', 'a') as f:
            print('F1 Score', f1_score[1], file=f)
            print('Accuracy', accuracy[1], file=f)

