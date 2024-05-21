import json
import os
import time
from random import shuffle
import numpy as np
import pandas as pd
import logging
import re

from owlready2 import default_world
from AutoCL.utils.ontology_acess import load_ontology, get_data_properties, get_object_properties
from sklearn.preprocessing import MultiLabelBinarizer

class OntologyTransformer:
    def __init__(self, onto, pos, neg):
        self.onto = onto
        self.pos = pos
        self.neg = neg
        self.column_headers = ['subject']

    def determine_label(self, sub):
        if sub in self.pos:
            return 1
        elif sub in self.neg:
            return 0
        else:
            return 2

    def transform_data_properties_to_tabular(self):
        properties_iri = []
        dict_of_triples_num = {}
        dict_of_triples_bool = {}

        data_prop = get_data_properties(self.onto)

        if data_prop is not None:
            properties_value = list(data_prop)
            for prop in properties_value:
                self.column_headers.append(prop.name)
            self.column_headers.append('label')

            for op in properties_value:
                if isinstance(op.range[0], int) or isinstance(op.range[0], float):
                    flag = 0
                elif isinstance(op.range[0], bool):
                    flag = 1
                else:
                    continue

                temp_prop_iri = op.iri
                properties_iri.append(temp_prop_iri)
                triples_list = list(default_world.sparql("""
                SELECT ?subject ?object
                WHERE {?subject <""" + str(temp_prop_iri) + """> ?object}
                """
                                                         )
                                    )

                for items in triples_list:
                    sub = items[0].iri
                    if flag:
                        obj = int(items[1])
                    else:
                        obj = items[1]

                    label = self.determine_label(sub)

                    if flag:
                        data_structure = dict_of_triples_bool
                    else:
                        data_structure = dict_of_triples_num

                    if sub not in data_structure:
                        data_structure[sub] = {'label': label, op.name: obj}
                    elif sub in data_structure:
                        data_structure[sub][op.name] = obj

            df_transposed_numeric_dtype = pd.DataFrame(dict_of_triples_num).transpose()
            df_transposed_categorical_dtype = pd.DataFrame(dict_of_triples_bool).transpose()

            self.save_to_csv(df_transposed_numeric_dtype, f"Numeric_data_properties_to_tabular_form.csv")
            self.save_to_csv(df_transposed_categorical_dtype, f"Boolean_data_properties_to_tabular_form.csv")

            return df_transposed_categorical_dtype, df_transposed_numeric_dtype

    def transform_object_properties_to_tabular(self):
        dict_of_triples = {}

        object_prop = get_object_properties(self.onto)

        if object_prop is not None:
            properties_value = list(object_prop)
            for prop in properties_value:
                self.column_headers.append(prop.name)
            self.column_headers.append('label')

            for op in properties_value:
                temp_prop_iri = op.iri
                triples_list = list(default_world.sparql("""
                SELECT ?subject ?object
                WHERE {?subject <""" + str(temp_prop_iri) + """> ?object}
                """
                                                         )
                                    )

                for items in triples_list:
                    sub = items[0].iri
                    obj = items[1].iri

                    label = self.determine_label(sub)

                    if sub not in dict_of_triples:
                        dict_of_triples[sub] = {'label': label, op.name: obj}
                    elif sub in dict_of_triples:
                        dict_of_triples[sub][op.name] = obj

            df_transposed_object_properties = pd.DataFrame(dict_of_triples).transpose()

            self.save_to_csv(df_transposed_object_properties, f"Object_properties_to_tabular_form.csv")

            return df_transposed_object_properties

    def one_hot_encoder(self, pd3):
        mlb = MultiLabelBinarizer(sparse_output=True)
        column_names = pd3.columns
        pd3 = pd3.replace(np.nan, '', regex=True)

        for cn in column_names[1:]:
            mlb.fit(pd3[cn])
            new_col_names = [cn + "_feature_name_%s" % c for c in mlb.classes_]
            pd3 = pd3.join(
                pd.DataFrame.sparse.from_spmatrix(
                    mlb.fit_transform(pd3.pop(cn)),
                    index=pd3.index,
                    columns=new_col_names
                )
            )
        return pd3

    def save_to_csv(self, dataframe, filename):
        # Get the directory of the current script
        current_script_dir = os.path.dirname(os.path.abspath(__file__))

        # Get the parent directory of the current script's directory
        parent_dir = os.path.abspath(os.path.join(current_script_dir, os.pardir))

        # Define the path to the results folder inside the parent directory
        results_folder = os.path.join(parent_dir, "results")

        # Ensure the results folder exists
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)

        # Define the full path for transformed graph to tabular format
        tf_file_path = os.path.join(results_folder, filename)
        if not dataframe.empty:
            dataframe.to_csv(tf_file_path)
