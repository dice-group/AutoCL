import os
from abc import ABC, abstractmethod
from ontolearn.knowledge_base import KnowledgeBase
from AutoCL.utils.ontology_acess import get_data_properties, get_object_properties
from owlready2 import destroy_entity

# TODO: Need to implement logger or use the already existing one
class FeatureSelectionStrategy(ABC):
    def __init__(self, onto, knowledgebase):
        self.onto = onto
        self.kb = knowledgebase


    @abstractmethod
    def select_features(self):
        pass

    # Private abstract method
    @abstractmethod
    def _initialize_feature_selection(self):
        pass

    def create_kb(self, features):
        """
        Creates a new knowledge base with selected features.

        Parameters:
        ----------
        features : list
            List of selected features.

        Returns:
        -------
        KnowledgeBase
            New knowledge base object with selected features.
        """
        # Get the directory of the current script
        current_script_dir = os.path.dirname(os.path.abspath(__file__))

        # Get the parent directory of the current script's directory
        parent_dir = os.path.abspath(os.path.join(current_script_dir, os.pardir))

        # Get object and data properties from the ontology
        prop_object = list(get_object_properties(self.onto)) + list(get_data_properties(self.onto))

        # Remove properties not present in the selected features list
        for prop in prop_object:
            if prop.name not in features:
                destroy_entity(prop)

        # Define the path to the results folder inside the parent directory
        results_folder = os.path.join(parent_dir, "results")

        # Ensure the results folder exists
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)

        # Define the full path for the saved ontology file
        ontology_file_path = os.path.join(results_folder, "kb_with_selected_features.owl")

        # Save the updated ontology in the results folder inside the parent directory
        self.onto.save(os.path.join(ontology_file_path))

        # Return the new knowledge base
        return KnowledgeBase(path=ontology_file_path)