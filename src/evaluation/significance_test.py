import click
import pandas as pd
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar
import csv
import pickle

from src.utils.tree_utils import TreeUtils


@click.command()
@click.option('--datasets1_path', help='First data set')
@click.option('--datasets2_path', help='Second data set')
@click.option('--path_to_tree', help='Path to tree')
def main(datasets1_path, datasets2_path, path_to_tree):

    file_ends = ['_5e-05_9.csv', '_5e-05_13.csv', '_5e-05_42.csv']
    datasets1 = {}
    datasets2 = {}
    for file_end in file_ends:

        datasets1[file_end] = pd.read_csv(datasets1_path + file_end, sep=';', encoding='utf-8', quotechar='"',
                                      quoting=csv.QUOTE_ALL)
        datasets2[file_end] = pd.read_csv(datasets2_path + file_end, sep=';', encoding='utf-8', quotechar='"',
                                      quoting=csv.QUOTE_ALL)

        # Augment data set
        datasets1[file_end] = augment_dataset(datasets1[file_end], path_to_tree)

        datasets1[file_end]['Correct_1'] = pd.Series(
            [row['Hierarchy Level 1 Label'] == row['Hierarchy Level 1 Prediction'] for index, row in
             datasets1[file_end].iterrows()])
        datasets2[file_end]['Correct_1'] = pd.Series(
            [row['Hierarchy Level 1 Label'] == row['Hierarchy Level 1 Prediction'] for index, row in
             datasets2[file_end].iterrows()])

        datasets1[file_end]['Correct_2'] = pd.Series(
            [row['Hierarchy Level 2 Label'] == row['Hierarchy Level 2 Prediction'] for index, row in
             datasets1[file_end].iterrows()])
        datasets2[file_end]['Correct_2'] = pd.Series(
            [row['Hierarchy Level 2 Label'] == row['Hierarchy Level 2 Prediction'] for index, row in
             datasets2[file_end].iterrows()])

        datasets1[file_end]['Correct_3'] = pd.Series(
            [row['Hierarchy Level 3 Label'] == row['Hierarchy Level 3 Prediction'] for index, row in
             datasets1[file_end].iterrows()])
        datasets2[file_end]['Correct_3'] = pd.Series(
            [row['Hierarchy Level 3 Label'] == row['Hierarchy Level 3 Prediction'] for index, row in
             datasets2[file_end].iterrows()])

    ds1_correct = aggregate_dataset_results(datasets1)
    ds2_correct = aggregate_dataset_results(datasets2)

    # mcnemar significance test to see if the results of the two classifiers are significantly different
    cl1_cor_cl2_cor = 0
    cl1_wr_cl2_cor = 0
    cl1_cor_cl2_wr = 0
    cl1_wr_cl2_wr = 0

    for ds1_record, ds2_record in zip(ds1_correct, ds2_correct):
        if ds1_record == True and ds2_record == True:
            cl1_cor_cl2_cor += 1
        elif ds1_record == False and ds2_record == True:
            cl1_wr_cl2_cor += 1
        elif ds1_record == True and ds2_record == False:
            cl1_cor_cl2_wr += 1
        elif ds1_record == False and ds2_record == False:
            cl1_wr_cl2_wr += 1

    cont_table = np.array([[cl1_cor_cl2_cor, cl1_cor_cl2_wr], [cl1_wr_cl2_cor, cl1_wr_cl2_wr]])
    print(cont_table)
    # https://machinelearningmastery.com/mcnemars-test-for-machine-learning/ (accessed on 29.07)
    # http://rasbt.github.io/mlxtend/user_guide/evaluate/mcnemar/
    if (cl1_cor_cl2_wr + cl1_wr_cl2_wr > 25):
        result = mcnemar(cont_table, correction=True)
    # result = mcnemar(cont_table, exact=False, correction=True)
    else:
        result = mcnemar(cont_table, exact=True)
    # result = mcnemar(cont_table, exact=True)
    # summarize the finding
    print('statistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))
    # interpret the p-value
    alphas = [0.05, 0.01]
    for alpha in alphas:
        print("For value alpha %f" % alpha)
        if result.pvalue > alpha:
            print('Same proportions of errors (fail to reject H0). Not significantly different.')
        else:
            print('Different proportions of errors (reject H0). Significantly different.')

def augment_dataset(dataset, path_to_tree):
    with open(path_to_tree, 'rb') as f:
        tree = pickle.load(f)
    treeUtils = TreeUtils(tree)

    augmented_dataset = pd.DataFrame()
    dataset['encoded_prediction'] = dataset['prediction'].apply(treeUtils.encode_node)
    dataset['encoded_category'] = dataset['category'].apply(treeUtils.encode_node)

    for index, row in dataset.iterrows():
        path_to_root_prediction = treeUtils.determine_path_to_root([row['encoded_prediction']])
        path_to_root_category = treeUtils.determine_path_to_root([row['encoded_category']])
        new_row = path_to_root_prediction + path_to_root_category
        df_new_row = pd.DataFrame([new_row], columns=['Hierarchy Level 1 Prediction', 'Hierarchy Level 2 Prediction', 'Hierarchy Level 3 Prediction',
                                       'Hierarchy Level 1 Label', 'Hierarchy Level 2 Label', 'Hierarchy Level 3 Label'])

        augmented_dataset = augmented_dataset.append(df_new_row, ignore_index=True)

    return augmented_dataset


def aggregate_dataset_results(datasets):
    ds_correct = []
    hierarchy_levels = ['Correct_1', 'Correct_2', 'Correct_3']
    for hierarchy_level in hierarchy_levels:
        for exp1_correct, exp2_correct, exp3_correct in zip(datasets['_5e-05_9.csv'][hierarchy_level].values,
                                                        datasets['_5e-05_13.csv'][hierarchy_level].values,
                                                        datasets['_5e-05_42.csv'][hierarchy_level].values):
            values = [exp1_correct, exp2_correct, exp3_correct]
            unique_values = set(values)

            if len(unique_values) == 1:
                ds_correct.append(unique_values.pop())
                continue

            count = 0
            for unique_value in unique_values:
                for value in values:
                    if value == unique_value:
                        count += 1
                if count >= 2:
                    ds_correct.append(unique_value)
                    break
                count = 0

    return ds_correct


if __name__ == '__main__':
    main()