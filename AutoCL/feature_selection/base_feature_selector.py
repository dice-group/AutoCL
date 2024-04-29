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
        # Get object and data properties from the ontology
        prop_object = list(get_object_properties(self.onto)) + list(get_data_properties(self.onto))

        # Remove properties not present in the selected features list
        for prop in prop_object:
            if prop.name not in features:
                destroy_entity(prop)

        # Save the updated ontology with selected features
        self.onto.save("kb_with_selected_features.owl")

        # Return the new knowledge base
        return KnowledgeBase(path="./kb_with_selected_features.owl")