import os
import time
from pathlib import Path

import optuna


class BaseOptimizer:
    def __init__(self, dataset, kb, lp, concept, val_pos, val_neg):
        self.study = None
        self.dataset = dataset
        self.kb = kb
        self.lp = lp
        self.concept = concept
        self.val_pos = val_pos
        self.val_neg = val_neg
        self.sampler = None  # Stores the chosen sampler
        self._df = None

    @property
    def df(self):
        """
        Property to access and set the dataframe.
        """
        return self._df

    @df.setter
    def df(self, value):
        """
        Setter for the dataframe.

        Args:
            value (pandas.DataFrame): The DataFrame to set.
        """
        self._df = value

    def optimize(self, objective_func, n_trials, direction='maximize'):
        """
        Performs hyperparameter optimization using the chosen objective function and sampler.

        Args:
            objective_func (callable): The objective function to be optimized.
            n_trials (int): The number of trials to run for optimization.
            direction: maximize or minimize
        """

        self.study = optuna.create_study(sampler=self.sampler, direction=direction)
        self.study.optimize(objective_func, n_trials=n_trials)

    def set_sampler(self, sampler):
        """
        Sets the sampler to be used for optimization.

        Args:
            sampler (optuna.samplers.Sampler): The sampler to be used.
        """

        self.sampler = sampler

    def save_results(self, df, trial=None):
        """
        Saves the optimization results to a CSV file.

        Args:
            df (pandas.DataFrame): The DataFrame containing the optimization results.
            trial (int, optional): The trial number (if applicable). Defaults to None.
        """
        # Get the directory of the current script
        current_script_dir = os.path.dirname(os.path.abspath(__file__))

        # Get the parent directory of the current script's directory
        parent_dir = os.path.abspath(os.path.join(current_script_dir, os.pardir))

        # Define the path to the results folder inside the parent directory
        results_folder = os.path.join(parent_dir, "results")

        # Ensure the results folder exists
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)

        timestr = str(time.strftime("%Y%m%d-%H%M%S"))
        lp_name = Path(self.dataset).stem
        filename = f'{lp_name}_Output_{trial if trial else ""}{timestr}'

        # Define the full path for hpo
        hpo_file_path = os.path.join(results_folder, filename)

        df.to_csv(hpo_file_path + '.csv', index=False)

    def get_best_hpo(self):
        if self.df is None or len(self.df) == 0:
            return None  # Return None if dataframe is empty or not set

        best_hpo = self.df.loc[self.df['Validation_f1_Score'] == self.df['Validation_f1_Score'].max()]
        if len(best_hpo.index) > 1:
            best_hpo = best_hpo.loc[(best_hpo['Validation_accuracy'] == best_hpo['Validation_accuracy'].max()) &
                                    (best_hpo['max_runtime'] == best_hpo['max_runtime'].min())]
        return best_hpo