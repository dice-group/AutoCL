import json
import os
import time
from random import shuffle
import logging
import re
import pandas as pd
from AutoCL.feature_selection.base_feature_selector import FeatureSelectionStrategy
from ontolearn.concept_learner import EvoLearner

from AutoCL.utils.ontology_acess import load_ontology, get_data_properties, get_object_properties
from owlapy.render import DLSyntaxObjectRenderer
from AutoCL.utils.ontology_transformer import OntologyTransformer as ot
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, SelectFpr
from sklearn.preprocessing import MultiLabelBinarizer
from scipy import sparse as sp
from sklearn.feature_selection import chi2, f_classif,  mutual_info_classif
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold



# TODO: define each attributes(Usage), check if return type is correct and implement logging
class EvolearnerBasedFeatureSelection(FeatureSelectionStrategy):
    """
    A class to perform feature selection using EvoLearner algorithm.

    Attributes:
    ----------
    path : str
        Path to the ontology file.
    lp : learning Problems
    knowledgebase : KnowledgeBase
        Knowledge base object containing the ontology.
    target_concepts : str
        Target concepts for feature selection.
    n : int, optional
        Initial hypothesis count from which features are to be extracted (default is 10).
    min_prop : int, optional
        Minimum number of properties to be selected (default is 5).
    max_hypo_count : int, optional
        Maximum number of hypothesis to be traversed (default is 800).
    """

    def __init__(self, path, lp, knowledgebase, target_concepts, n=10, min_prop=5, max_hypo_count=800):
        """
        Initializes the EvolearnerBasedFeatureSelection class.

        Parameters:
        ----------
        path : str
            Path to the ontology file.
        lp :
            Learning Problems
        knowledgebase : KnowledgeBase
            Knowledge base object containing the ontology.
        target_concepts : str
            Target concepts for feature selection.
        n : int, optional
            Initial hypothesis count from which features are to be extracted (default is 10).
        min_prop : int, optional
            Minimum number of properties to be selected (default is 1).
        max_hypo_count : int, optional
            Maximum number of hypothesis to be traversed (default is 800).
        """
        # Initialize instance variables
        self.new_kb = None
        self.features = None
        self.lp = lp
        self.kb = knowledgebase
        self.target_concept = target_concepts
        self.min_prop = min_prop
        self.n = n
        self.max_hypo_count = max_hypo_count

        # Load ontology from the provided path
        self.onto = load_ontology(path)

        # Initialize feature selection
        self._initialize_feature_selection()

    def _initialize_feature_selection(self):
        """
        Private method to initialize feature selection using EvoLearner.
        """
        self.f_evo()

    def f_evo(self):
        """
        Executes the EvoLearner-based feature selection process.
        """
        self.features = self.select_features()
        self.new_kb = self.create_kb(self.features)

    def select_features(self):
        """
        Selects the most relevant features using EvoLearner.

        Returns:
        -------
        list
            A list of selected feature names based on the EvoLearner model.
        """
        # Initialize variables
        current_hypothesis_count = self.n
        properties_from_concepts = []

        # Initialize EvoLearner model with the provided knowledge base
        model = EvoLearner(knowledge_base=self.kb)
        model.fit(self.lp)

        # Counter to track repeated property selections
        counter = 0
        previous_properties_list = []

        # Get object and data properties from the ontology
        prop_object = list(get_object_properties(self.onto)) + list(get_data_properties(self.onto))

        # Loop until minimum properties are selected or maximum hypothesis count is reached
        while (len(properties_from_concepts) < self.min_prop and
               current_hypothesis_count <= self.max_hypo_count and
               counter < 100):

            # Retrieve the best hypotheses from the EvoLearner model
            hypotheses = list(model.best_hypotheses(n=current_hypothesis_count))

            # Renderer to convert DL syntax to readable format
            dlr = DLSyntaxObjectRenderer()
            concept_sorted = [dlr.render(c.concept) for c in hypotheses]

            # Extract properties present in the sorted concepts
            properties = []
            for concepts in concept_sorted:
                data_object_properties_in_concepts = [prop.name for prop in prop_object if prop.name in concepts]
                properties.extend(data_object_properties_in_concepts)

            # Remove duplicates from the list of properties
            properties_from_concepts = list(set(properties))

            # Check for repeated property selections
            if properties_from_concepts == previous_properties_list:
                counter += 1
            else:
                counter = 0

            # Update the previous properties list
            previous_properties_list = properties_from_concepts.copy()

            # Increment the hypothesis count for the next iteration
            current_hypothesis_count += 1

        # Return the selected properties
        return properties_from_concepts


# TO DO: Add doc string, comments and implement logging
class TableBasedFeatureSelection(FeatureSelectionStrategy):
    """
        Class for performing table-based feature selection.

        Args:
            path (str): The path to the ontology.
            lp (LogicProgram): The logic program.
            knowledgebase (KnowledgeBase): The knowledge base.
            target_concepts (list): List of target concepts.
            pos (list): List of positive examples.
            neg (list): List of negative examples.
    """
    def __init__(self, path, lp, knowledgebase, target_concepts, pos, neg):
        # Initialize instance variables
        self.new_kb = None
        self.features = None
        self.lp = lp
        self.kb = knowledgebase
        self.target_concept = target_concepts
        self.pos = pos
        self.neg = neg

        # Load ontology from the provided path
        self.onto = load_ontology(path)

        # Initialize feature selection
        self._initialize_feature_selection()

    def _initialize_feature_selection(self):
        """
        Private method to initialize feature selection using EvoLearner.
        """
        obj_prop, bool_prop, numeric_prop = self._preprocess()
        self.features = self.select_features(obj_prop, bool_prop, numeric_prop)
        self.new_kb = self.create_kb(self.features)

    def _preprocess(self):
        transformer_obj = ot(self.onto, self.pos, self.neg)
        obj_prop = transformer_obj.transform_object_properties_to_tabular()
        bool_dtype_df, numeric_dtype_df = transformer_obj.transform_data_properties_to_tabular()

        # Perform one-hot encoding for object properties if not empty
        pd_one_hot = pd.DataFrame()
        if not obj_prop.empty:
            pd_one_hot = transformer_obj.one_hot_encoder(obj_prop)

        return pd_one_hot, bool_dtype_df, numeric_dtype_df

    def select_k_best_features(self, data_frame, method, k=5):
        '''
        Select K-Best features based on chi2/mutual_info_classif analysis.

        Parameters:
        - data_frame: The input DataFrame.
        - method: The method to use for feature selection (e.g., chi2, mutual_info_classif).
        - k: The desired number of features to select.

        Returns:
        A list of selected feature names.
        '''

        try:
            # Check if the DataFrame is empty
            if data_frame.empty:
                logging.warning("Input DataFrame is empty.")
                return []

            X = data_frame.iloc[:, 1:].values
            labels = data_frame.iloc[:, 0].values
            k_best_features_names = set()

            iteration = 0
            while len(k_best_features_names) < k and iteration < 100:
                select_k_best_classifier = SelectKBest(method, k=k)
                select_k_best_classifier.fit(X, labels.tolist())
                mask = select_k_best_classifier.get_support(indices=True)

                # Convert indices to column names
                current_k_best_features_names = set(data_frame.columns[1:][mask])
                # Extract only the part before "_feature_name" when "_feature_name" is present
                current_k_best_features_names = {
                    re.sub(r'_feature_name.*', '', feature) if '_feature_name' in feature else feature
                    for feature in current_k_best_features_names
                }

                if current_k_best_features_names != k_best_features_names:
                    k_best_features_names = current_k_best_features_names
                    iteration = 0  # Reset iteration counter
                else:
                    iteration += 1
                    k += 1

                # Handle the case where k is greater than the number of available features
                if k > len(data_frame.columns[1:]):
                    k = 'all'
                    break

            return list(k_best_features_names)

        except Exception as e:
            logging.error(e)
            return []

    def select_features(self, obj_prop, bool_prop, numeric_prop):

        # Feature selection for object properties
        features_object_properties = []

        # Feature selection for data properties
        features_data_properties_categorical = []
        feature_data_prop_numeric = []

        if not obj_prop.empty:
            features_object_properties = list(self.select_k_best_features(obj_prop, chi2))

        # Check if the total selected features are less than 5
        total_selected_features = len(features_object_properties)

        if total_selected_features < 5:
            if not bool_prop.empty:
                features_data_properties_categorical = self.select_k_best_features(bool_prop, chi2)

            if not numeric_prop.empty:
                feature_data_prop_numeric = self.select_k_best_features(numeric_prop, chi2)

            features = (
                    features_object_properties + feature_data_prop_numeric + features_data_properties_categorical
            )
        else:
            features = features_object_properties
        return features