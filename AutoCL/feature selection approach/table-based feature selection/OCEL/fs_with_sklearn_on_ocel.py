import json
import os
import time
from random import shuffle
import numpy as np
import pandas as pd
import logging
import re

from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, SelectFpr
from sklearn.preprocessing import MultiLabelBinarizer
from scipy import sparse as sp
from sklearn.feature_selection import chi2, f_classif,  mutual_info_classif
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from owlready2 import get_ontology, destroy_entity, default_world

from examples.search import calc_prediction
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.owlapy.model import OWLNamedIndividual, IRI
from wrapper_ocel import OcelWrapper


DIRECTORY = './ICML/FINAL/FS_WITH_SKLEARN_ON_OCEL/'
LOG_FILE = 'featureSelectionWithSklearn.log'
DATASET = 'mammograph'
PATH = f'dataset/{DATASET}.json'

logging.basicConfig(filename=LOG_FILE,
                    filemode="a",
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)

try:
    os.chdir("Ontolearn/examples")
except FileNotFoundError:
    logging.error(FileNotFoundError)
    pass


with open(PATH) as json_file:
    settings = json.load(json_file)
if not os.path.exists(DIRECTORY):
    os.makedirs(DIRECTORY)


def get_data_properties(onto):
    try:
        data_properties = onto.data_properties()
    except Exception as e:
        data_properties = None
        logging.error(e)
    return data_properties


def get_object_properties(onto):
    try:
        object_properties = onto.object_properties()
    except Exception as e:
        object_properties = None
        logging.error(e)
    return object_properties


def calc_variance_threshold(pd3):
    pd3 = (pd3.notnull()).astype('int')
    sparse_matrix = sp.csr_matrix.tocsr(pd3.iloc[:, 1:].values)
    pd3.to_csv(f"{DATASET}_dataset_converted_to_bool.csv")

    try:
        sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
        output_variance_threshold = sel.fit_transform(sparse_matrix)
        output_variance_threshold.to_csv(f"{DATASET}_best_features_variance_threshold.csv")
    except ValueError:
        logging.error("None of the Features meet the minimum variance threshold")


def select_k_best_features(data_frame, method, k=5):
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

            # print('Current F', current_k_best_features_names)
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


def select_fpr_features(pd3, method):
    '''
    Select features based on chi2/f_classif analysis
    False Positive Rate test checks the total amt. of False detection
    '''
    X = pd3.iloc[:, 1:].values
    labels = pd3.iloc[:, 0].values

    try:
        select_k_best_classifier = SelectFpr(
                                method, alpha=0.01).fit(
                                X, labels.tolist())
        mask = select_k_best_classifier.get_support(indices=True)
        fpr_features = pd.DataFrame(index=pd3.index)
        fpr_features = pd3.iloc[:, [x+1 for x in mask.tolist()]]
        fpr_features.to_csv(f"{DATASET}_fpr_features_chi2.csv")
        fpr_features_names = fpr_features.columns

    except Exception as e:
        logging.error(e)
        fpr_features_names = []
    return fpr_features_names


def one_hot_encoder(pd3):
    mlb = MultiLabelBinarizer(sparse_output=True)
    column_names = pd3.columns
    pd3 = pd3.replace(np.nan, '', regex=True)

    for cn in column_names[1:]:
        mlb.fit(pd3[cn])
        new_col_names = [cn+"_feature_name_%s" % c for c in mlb.classes_]
        pd3 = pd3.join(
            pd.DataFrame.sparse.from_spmatrix(
                mlb.fit_transform(pd3.pop(cn)),
                index=pd3.index,
                columns=new_col_names)
            )
    return pd3


def create_new_kb_without_features(features):
    prop_object = list(get_object_properties(onto)) + list(get_data_properties(onto))

    for prop in prop_object:
        if prop.name not in features:
            # print('prop.name', prop.name)
            destroy_entity(prop)
    onto.save("kb_with_selected_features.owl")
    return KnowledgeBase(path="./kb_with_selected_features.owl")


def iterate_object_properties(properties, pos, neg):
    '''
    Iterate over properties and convert it to tabular data
    '''
    properties_iri = []
    dict_of_triples = {}
    column_headers = ['subject']
    if properties is not None:
        properties_value = list(properties)
        for prop in properties_value:
            column_headers.append(prop.name)
        column_headers.append('label')

        for op in properties_value:
            temp_prop_iri = op.iri
            properties_iri.append(temp_prop_iri)
            triples_list = list(default_world.sparql("""
            SELECT ?subject ?object
            WHERE {?subject <""" + str(temp_prop_iri) + """> ?object}
            """))

            for items in triples_list:
                sub, obj = items[0].iri, items[1].iri
                if sub in pos:
                    label = 1
                elif sub in neg:
                    label = 0
                else:
                    label = 2

                if sub not in dict_of_triples:
                    dict_of_triples[sub] = {'label': label, op.name: [obj]}
                elif sub in dict_of_triples:
                    if op.name in dict_of_triples[sub].keys():
                        dict_of_triples[sub][op.name].append(obj)
                    else:
                        dict_of_triples[sub][op.name] = [obj]

        pandas_dataframe = pd.DataFrame(dict_of_triples)
        df_transposed = pandas_dataframe.transpose()
        df_transposed.to_csv(f"{DATASET}_Object_properties_to_tabular_form.csv")
    else:
        df_transposed = pd.DataFrame()
    return df_transposed


def transform_object_properties(obj_prop, pos, neg):
    '''
    Read Object properties and call
    iterate properties to convert to tabular form
    '''
    with onto:
        obj_prop = get_object_properties(onto)
        df_object_prop = iterate_object_properties(obj_prop, pos, neg)
        return df_object_prop


def transform_data_properties(onto, pos, neg):
    '''
    Read data properties and and call iterate
    properties to convert to tabular form
    '''
    with onto:
        data_prop = get_data_properties(onto)

        properties_iri = []
        dict_of_triples_num = {}
        dict_of_triples_bool = {}
        column_headers = ['subject']

        if data_prop is not None:
            properties_value = list(data_prop)
            for prop in properties_value:
                column_headers.append(prop.name)
            column_headers.append('label')

            for op in properties_value:
                if (op.range[0] is int):
                    flag = 0
                elif (op.range[0] is float):
                    flag = 0
                elif (op.range[0] is bool):
                    flag = 1
                else:
                    continue

                temp_prop_iri = op.iri
                properties_iri.append(temp_prop_iri)
                triples_list = list(default_world.sparql("""
                SELECT ?subject ?object
                WHERE {?subject <""" + str(temp_prop_iri) + """> ?object}
                """))

                for items in triples_list:
                    sub = items[0].iri
                    if flag:
                        obj = int(items[1])
                    else:
                        obj = items[1]

                    if sub in pos:
                        label = 1
                    elif sub in neg:
                        label = 0
                    else:
                        label = 2

                    if flag:
                        data_structure = dict_of_triples_bool
                    else:
                        data_structure = dict_of_triples_num

                    if sub not in data_structure:
                        data_structure[sub] = {'label': label, op.name: obj}
                    elif sub in data_structure:
                        data_structure[sub][op.name] = obj

            pandas_dataframe_numeric_data_types = pd.DataFrame(dict_of_triples_num)
            pandas_dataframe_categorical_data_type = pd.DataFrame(dict_of_triples_bool)
            df_transposed_numeric_dtype = pandas_dataframe_numeric_data_types.transpose()
            df_transposed_categorical_dtype = pandas_dataframe_categorical_data_type.transpose()

            df_transposed_numeric_dtype.to_csv(f"{DATASET}_Numeric_data_properties_to_tabular_form.csv")
            df_transposed_categorical_dtype.to_csv(f"{DATASET}_Boolean_data_properties_to_tabular_form.csv")

        if not df_transposed_numeric_dtype.empty:
            df_transposed_numeric_dtype = df_transposed_numeric_dtype.replace(np.nan, 0, regex=True)
        if not df_transposed_categorical_dtype.empty:
            df_transposed_categorical_dtype = df_transposed_categorical_dtype.replace(np.nan, 0, regex=True)

        return df_transposed_categorical_dtype, df_transposed_numeric_dtype


if __name__ == "__main__":
    onto = get_ontology(settings['data_path']).load()
    kb = KnowledgeBase(path=settings['data_path'])

    for str_target_concept, examples in settings['problems'].items():
        p = set(examples['positive_examples'])
        n = set(examples['negative_examples'])

        typed_pos = list(set(map(OWLNamedIndividual, map(IRI.create, p))))
        typed_neg = list(set(map(OWLNamedIndividual, map(IRI.create, n))))

        # Shuffle the Positive and Negative Sample
        shuffle(typed_pos)
        shuffle(typed_neg)

        # Split the data into Training Set and Test Set
        train_pos = set(typed_pos[:int(len(typed_pos) * 0.8)])
        train_neg = set(typed_neg[:int(len(typed_neg) * 0.8)])
        test_pos = set(typed_pos[-int(len(typed_pos) * 0.2):])
        test_neg = set(typed_neg[-int(len(typed_neg) * 0.2):])
        lp = PosNegLPStandard(pos=train_pos, neg=train_neg)

        st = time.time()

        # Transform properties
        object_properties_df = transform_object_properties(onto, p, n)
        bool_dtype_df, numeric_dtype_df = transform_data_properties(onto, p, n)

        # Feature selection for object properties
        features_object_properties = []

        if not object_properties_df.empty:
            pd_one_hot = one_hot_encoder(object_properties_df)
            st1 = time.time()
            features_object_properties = list(select_k_best_features(pd_one_hot, chi2))

        # Feature selection for data properties
        features_data_properties_categorical = []
        feature_data_prop_numeric = []

        # Check if the total selected features are less than 5
        total_selected_features = len(features_object_properties)

        if total_selected_features < 5:
            if not bool_dtype_df.empty:
                features_data_properties_categorical = select_k_best_features(bool_dtype_df, chi2)

            if not numeric_dtype_df.empty:
                feature_data_prop_numeric = select_k_best_features(numeric_dtype_df, mutual_info_classif)

            logging.info(f"OBJECT_PROP: {features_object_properties}")
            logging.info(f"DATA_PROP_Numeric: {feature_data_prop_numeric}")
            logging.info(f"DATA_PROP_BOOL: {features_data_properties_categorical}")

            selected_properties = (
                features_object_properties + feature_data_prop_numeric + features_data_properties_categorical
            )
        else:
            selected_properties = features_object_properties

        et = time.time()
        elapsed_time = et - st
        elapsed_time_1 = et - st1

        # Create a new KB
        kb_with_selected_features = create_new_kb_without_features(selected_properties)

        # Train the model
        st2 = time.time()
        wrap_obj = OcelWrapper(knowledge_base=kb_with_selected_features,
                               max_runtime=600,
                               max_num_of_concepts_tested=10_000_000_000,
                               iter_bound=10_000_000_000)
        model = wrap_obj.get_ocel_model()
        model.fit(lp, verbose=False)
        model.save_best_hypothesis(n=3, path='Predictions_{0}'.
                                   format(str_target_concept))
        hypotheses = list(model.best_hypotheses(n=1))
        hypotheses = [hypo for hypo in hypotheses]
        print(hypotheses)
        predictions = model.predict(individuals=list(test_pos | test_neg),
                                    hypotheses=hypotheses
                                    )
        f1_score, accuracy = calc_prediction(predictions, test_pos, test_neg)
        quality = hypotheses[0].quality

        et2 = time.time()
        elapsed_time_2 = et2 - st2
        with open(f'{DIRECTORY}{DATASET}_featureSelection.txt', 'a') as f:
            print('--------str_target_concept-----------', str_target_concept, file=f)
            print('F1 Score', f1_score[1], file=f)
            print('Accuracy', accuracy[1], file=f)
            print('selected feature', selected_properties, file=f)
            print('Time Taken for FS With One hot', elapsed_time, file=f)
            print('Time Taken Without one hot', elapsed_time_1, file=f)
            print('Time Taken For Fit and Predict', elapsed_time_2, file=f)
