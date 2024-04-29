from owlready2 import get_ontology

# TODO: Implement Logging

def load_ontology(ontology_path):
  """
  Loads an ontology from the provided path using OWLReady2.

  Args:
      ontology_path (str): Path to the ontology file (e.g., .owl).

  Returns:
      OWLAlchemy.Ontology: The loaded ontology object.
  """
  try:
    return get_ontology(ontology_path).load()
  except Exception as e:
    # TODO: Log e
    return None

# TODO: Log e
def get_data_properties(onto):
    try:
        data_properties = onto.data_properties()
    except Exception as e:
        data_properties = None
    return data_properties

# TODO: Log e
def get_object_properties(onto):
    try:
        object_properties = onto.object_properties()
    except Exception as e:
        object_properties = None
    return object_properties