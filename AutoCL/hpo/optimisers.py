import pandas as pd
from AutoCL.hpo.base_optimiser import BaseOptimizer
from AutoCL.utils.wrapper import CeloeWrapper, OcelWrapper, EvoLearnerWrapper
from AutoCL.utils.search import calc_prediction


class EvolearnerOptimizer(BaseOptimizer):
    def __init__(self, dataset, kb, lp, concept, val_pos, val_neg):
        super().__init__(dataset, kb, lp, concept, val_pos, val_neg)
        self.df = pd.DataFrame(columns=['LP', 'max_runtime', 'tournament_size',
                                        'height_limit', 'card_limit',
                                        'use_data_properties', 'quality_func',
                                        'use_inverse_prop', 'value_splitter',
                                        'quality_score', 'Validation_f1_Score',
                                        'Validation_accuracy'])

    def objective_with_categorical_distribution(self, trial):
        # categorical distribution efficiently works for the following samplers
        # Random Sampler, Grid Sampler, TPE Sampler and NSGAII Sampler
        max_runtime = trial.suggest_int("max_runtime", 2, 20)
        tournament_size = trial.suggest_int("tournament_size", 2, 20)
        height_limit = trial.suggest_int('height_limit', 3, 25)
        card_limit = trial.suggest_int('card_limit', 5, 10)
        use_data_properties = trial.suggest_categorical('use_data_properties',
                                                        ['True', 'False'])
        use_inverse = trial.suggest_categorical('use_inverse', ['True', 'False'])
        quality_func = trial.suggest_categorical('quality_func', ['F1', 'Accuracy'])
        value_splitter = trial.suggest_categorical('value_splitter',
                                                   ['binning_value_splitter',
                                                    'entropy_value_splitter'])
        # call the wrapper class
        wrap_obj = EvoLearnerWrapper(knowledge_base=self.kb,
                                     max_runtime=max_runtime,
                                     tournament_size=tournament_size,
                                     height_limit=height_limit,
                                     card_limit=card_limit,
                                     use_data_properties=use_data_properties,
                                     use_inverse=use_inverse,
                                     quality_func=quality_func,
                                     value_splitter=value_splitter)
        model = wrap_obj.get_evolearner_model()
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
        space['tournament_size'] = tournament_size
        space['height_limit'] = height_limit
        space['card_limit'] = card_limit
        space['use_data_properties'] = use_data_properties
        space['use_inverse'] = use_inverse
        space['value_splitter'] = value_splitter
        space['quality_func'] = quality_func
        space['quality_score'] = quality
        space['Validation_f1_Score'] = f1_score[1]
        space['Validation_accuracy'] = accuracy[1]
        self.__save_df(**space)
        return quality

    def objective_without_categorical_distribution(self, trial):
        # Categorical distribution does not work for QMC and CMEs sampler
        # The categorical distribution values are sampled in such cases
        # using Random Sampler
        # Using a threshold value to categorise the hyperparameters.
        max_runtime = trial.suggest_int("max_runtime", 2, 20)
        tournament_size = trial.suggest_int("tournament_size", 2, 10)
        height_limit = trial.suggest_int('height_limit', 3, 25)
        card_limit = trial.suggest_int('card_limit', 5, 10)
        use_data_properties = trial.suggest_int('use_data_properties', 1, 2)
        use_inverse = trial.suggest_int('use_inverse', 1, 2)
        quality_func = trial.suggest_int('quality_func', 1, 2)
        value_splitter = trial.suggest_int('value_splitter', 1, 2)

        use_data_properties = 'True' if use_data_properties >= 2 else 'False'
        use_inverse = 'True' if use_inverse >= 2 else 'False'
        quality_func = 'F1' if quality_func >= 2 else 'Accuracy'
        value_splitter = 'binning_value_splitter' if value_splitter >= 2 else 'entropy_value_splitter'

        # call the wrapper class
        wrap_obj = EvoLearnerWrapper(knowledge_base=self.kb,
                                     max_runtime=max_runtime,
                                     tournament_size=tournament_size,
                                     height_limit=height_limit,
                                     card_limit=card_limit,
                                     use_data_properties=use_data_properties,
                                     use_inverse=use_inverse,
                                     quality_func=quality_func,
                                     value_splitter=value_splitter)
        model = wrap_obj.get_evolearner_model()
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
        space['tournament_size'] = tournament_size
        space['height_limit'] = height_limit
        space['card_limit'] = card_limit
        space['use_data_properties'] = use_data_properties
        space['use_inverse'] = use_inverse
        space['value_splitter'] = value_splitter
        space['quality_func'] = quality_func
        space['quality_score'] = quality
        space['Validation_f1_Score'] = f1_score[1]
        space['Validation_accuracy'] = accuracy[1]
        self.save_df(**space)
        return quality

    def __save_df(self, **space):
        self.df.loc[len(self.df.index)] = [self.concept,
                                           space['max_runtime'],
                                           space['tournament_size'],
                                           space['height_limit'],
                                           space['card_limit'],
                                           space['use_data_properties'],
                                           space['quality_func'],
                                           space['use_inverse'],
                                           space['value_splitter'],
                                           space['quality_score'],
                                           space['Validation_f1_Score'],
                                           space['Validation_accuracy']]

    def save_results(self, trial=None):
        super().save_results(self.df, trial)



class CeloeOptimizer(BaseOptimizer):
    def __init__(self, dataset, kb, lp, concept, val_pos, val_neg):
        super().__init__(dataset, kb, lp, concept, val_pos, val_neg)
        self.df = pd.DataFrame(columns=['LP', 'max_runtime',
                                        'max_num_of_concepts_tested', 'iter_bound',
                                        'quality_func',
                                        'quality_score', 'Validation_f1_Score',
                                        'Validation_accuracy'])

    def objective_with_categorical_distribution(self, trial):
        max_runtime = trial.suggest_int("max_runtime", 1, 600)
        max_num_of_concepts_tested = trial.suggest_int("max_num_of_concepts_tested", 2, 10_000_000_00)
        iter_bound = trial.suggest_int('iter_bound', 2, 10_000_000_00)
        quality_func = trial.suggest_categorical('quality_func', ['F1', 'Accuracy'])

        wrap_obj = CeloeWrapper(knowledge_base=self.kb,
                                max_runtime=max_runtime,
                                max_num_of_concepts_tested=max_num_of_concepts_tested,
                                iter_bound=iter_bound,
                                quality_func=quality_func,
                                )
        model = wrap_obj.get_celoe_model()
        model.fit(self.lp, verbose=False)
        model.save_best_hypothesis(n=3, path=f'Predictions_{self.concept}')
        hypotheses = list(model.best_hypotheses(n=1))
        predictions = model.predict(individuals=list(self.val_pos | self.val_neg),
                                    hypotheses=hypotheses)
        f1_score, accuracy = calc_prediction(predictions, self.val_pos, self.val_neg)
        quality = hypotheses[0].quality

        space = dict()
        space['max_runtime'] = max_runtime
        space['max_num_of_concepts_tested'] = max_num_of_concepts_tested
        space['iter_bound'] = iter_bound
        space['quality_func'] = quality_func
        space['quality_score'] = quality
        space['Validation_f1_Score'] = f1_score[1]
        space['Validation_accuracy'] = accuracy[1]
        self.__save_df(**space)

        return quality

    def objective_without_categorical_distribution(self, trial):
        max_runtime = trial.suggest_int("max_runtime", 2, 500)
        max_num_of_concepts_tested = trial.suggest_int("max_num_of_concepts_tested", 2, 10_000_000_000)
        iter_bound = trial.suggest_int('iter_bound', 3, 10_000_000_000)

        quality_func = trial.suggest_int('quality_func', 1, 2)
        quality_func = 'F1' if quality_func >= 2 else 'Accuracy'

        wrap_obj = CeloeWrapper(knowledge_base=self.kb,
                                max_runtime=max_runtime,
                                max_num_of_concepts_tested=max_num_of_concepts_tested,
                                iter_bound=iter_bound,
                                quality_func=quality_func,
                                )
        model = wrap_obj.get_celoe_model()
        model.fit(self.lp, verbose=False)
        model.save_best_hypothesis(n=3, path=f'Predictions_{self.concept}')
        hypotheses = list(model.best_hypotheses(n=1))
        predictions = model.predict(individuals=list(self.val_pos | self.val_neg),
                                    hypotheses=hypotheses)
        f1_score, accuracy = calc_prediction(predictions, self.val_pos, self.val_neg)
        quality = hypotheses[0].quality

        space = dict()
        space['max_runtime'] = max_runtime
        space['max_num_of_concepts_tested'] = max_num_of_concepts_tested
        space['iter_bound'] = iter_bound
        space['quality_func'] = quality_func
        space['quality_score'] = quality
        space['Validation_f1_Score'] = f1_score[1]
        space['Validation_accuracy'] = accuracy[1]
        self.__save_df(**space)

        return quality

    def __save_df(self, **space):
        self.df.loc[len(self.df.index)] = [self.concept, space['max_runtime'],
                                           space['max_num_of_concepts_tested'], space['iter_bound'],
                                           space['quality_func'],
                                           space['quality_score'],
                                           space['Validation_f1_Score'], space['Validation_accuracy']]

    def save_results(self, trial=None):
        super().save_results(self.df, trial)


class OcelOptimizer(BaseOptimizer):
    def __init__(self, dataset, kb, lp, concept, val_pos, val_neg):
        super().__init__(dataset, kb, lp, concept, val_pos, val_neg)
        self.df = pd.DataFrame(columns=['LP', 'max_runtime',
                                        'max_num_of_concepts_tested', 'iter_bound',
                                        'quality_func',
                                        'quality_score', 'Validation_f1_Score',
                                        'Validation_accuracy'])

    def objective_with_categorical_distribution(self, trial):
        max_runtime = trial.suggest_int("max_runtime", 1, 600)
        max_num_of_concepts_tested = trial.suggest_int("max_num_of_concepts_tested", 2, 10_000_000_00)
        iter_bound = trial.suggest_int('iter_bound', 2, 10_000_000_00)
        quality_func = trial.suggest_categorical('quality_func', ['F1', 'Accuracy'])

        wrap_obj = OcelWrapper(knowledge_base=self.kb,
                                max_runtime=max_runtime,
                                max_num_of_concepts_tested=max_num_of_concepts_tested,
                                iter_bound=iter_bound,
                                quality_func=quality_func,
                                )
        model = wrap_obj.get_ocel_model()
        model.fit(self.lp, verbose=False)
        model.save_best_hypothesis(n=3, path=f'Predictions_{self.concept}')
        hypotheses = list(model.best_hypotheses(n=1))
        predictions = model.predict(individuals=list(self.val_pos | self.val_neg),
                                    hypotheses=hypotheses)
        f1_score, accuracy = calc_prediction(predictions, self.val_pos, self.val_neg)
        quality = hypotheses[0].quality

        space = dict()
        space['max_runtime'] = max_runtime
        space['max_num_of_concepts_tested'] = max_num_of_concepts_tested
        space['iter_bound'] = iter_bound
        space['quality_func'] = quality_func
        space['quality_score'] = quality
        space['Validation_f1_Score'] = f1_score[1]
        space['Validation_accuracy'] = accuracy[1]
        self.__save_df(**space)

        return quality

    def objective_without_categorical_distribution(self, trial):
        max_runtime = trial.suggest_int("max_runtime", 2, 500)
        max_num_of_concepts_tested = trial.suggest_int("max_num_of_concepts_tested", 2, 10_000_000_000)
        iter_bound = trial.suggest_int('iter_bound', 3, 10_000_000_000)

        quality_func = trial.suggest_int('quality_func', 1, 2)
        quality_func = 'F1' if quality_func >= 2 else 'Accuracy'

        wrap_obj = OcelWrapper(knowledge_base=self.kb,
                                max_runtime=max_runtime,
                                max_num_of_concepts_tested=max_num_of_concepts_tested,
                                iter_bound=iter_bound,
                                quality_func=quality_func,
                                )
        model = wrap_obj.get_ocel_model()
        model.fit(self.lp, verbose=False)
        model.save_best_hypothesis(n=3, path=f'Predictions_{self.concept}')
        hypotheses = list(model.best_hypotheses(n=1))
        predictions = model.predict(individuals=list(self.val_pos | self.val_neg),
                                    hypotheses=hypotheses)
        f1_score, accuracy = calc_prediction(predictions, self.val_pos, self.val_neg)
        quality = hypotheses[0].quality

        space = dict()
        space['max_runtime'] = max_runtime
        space['max_num_of_concepts_tested'] = max_num_of_concepts_tested
        space['iter_bound'] = iter_bound
        space['quality_func'] = quality_func
        space['quality_score'] = quality
        space['Validation_f1_Score'] = f1_score[1]
        space['Validation_accuracy'] = accuracy[1]
        self.__save_df(**space)

        return quality

    def __save_df(self, **space):
        self.df.loc[len(self.df.index)] = [self.concept, space['max_runtime'],
                                           space['max_num_of_concepts_tested'], space['iter_bound'],
                                           space['quality_func'],
                                           space['quality_score'],
                                           space['Validation_f1_Score'], space['Validation_accuracy']]

    def save_results(self, trial=None):
        super().save_results(self.df, trial)