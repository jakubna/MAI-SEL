from decisionforest import DecisionForest
from randomforest import RandomForest

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import argparse
import numpy as np
import math
import os

def percentage_type(x):
    x = float(x)
    if not 0.0 <= x <= 1.0:
        raise argparse.ArgumentTypeError("Value must be between 0.0 and 1.0")
    return x

parser = argparse.ArgumentParser('main')
parser.add_argument('-a', '--algorithm',
                    help='Decision forest [df] or random forest [rf] algorithm. Possible values: [df, rf]', type=str)
parser.add_argument('-d', '--dataset_size',
                    help='Size of the dataset. Possible values: [small, medium, large]', type=str)
parser.add_argument('-t', '--test_percentage',
                    help='Percentage of the dataset that will be used for testing. Possible values: [between 0.0 and 1.0] Default: 0.2',
                    type=percentage_type, default=0.2)
args = parser.parse_args()

if args.dataset_size not in ['small', 'medium', 'large', "demo"]:
    raise Exception("Wrong dataset size. Possible values: [small, medium, large, demo]")
if args.algorithm not in ['df', 'rf']:
    raise Exception("Wrong algorithm value. Possible values: [df, rf]")

def preprocess_data(file_path, test_size):
    df = pd.read_csv(file_path, sep=',')
    headers = list(df.columns)
    headers.remove('class')
    train_data, test_data = train_test_split(df, test_size=test_size, random_state=42)
    return train_data, test_data, headers
file_path = ''
demo = False

if args.dataset_size == "small":
    file_path = '../Data/iris.data'
elif args.dataset_size == "medium":
    file_path = '../Data/car.data'
elif args.dataset_size == "large":
    file_path = '../Data/nursery.data'
elif args.dataset_size == "demo":
    file_path = '../Data/human-identify.data'
    demo = True

if demo:
    df = pd.read_csv(file_path, sep=',')
    headers = list(df.columns)
    headers.remove('class')
    M = len(headers)
    if args.algorithm == 'df':
        forest = DecisionForest(number_features=M, number_trees=5)
    elif args.algorithm == 'rf':
        forest = RandomForest(number_features=M, number_trees=5)
    forest_results  = forest.make_and_test_forest(df, df)
    print('Accuracy: {}'.format(forest.accuracy))
    print('Class prediction')
    print(forest.predictions)
    print('Multiple class prediction')
    print(forest.multiple_predictions)
    print('The last created tree')
    print(forest.forest)
else:
    train_data, test_data, headers = preprocess_data(file_path, args.test_percentage)

    print("Name of dataset: {}".format(file_path[7:-5]))
    print("Number of training instances: {}".format(train_data.shape[0]))
    print("Number of test instances: {}".format(test_data.shape[0]))

    ### VARIABLES
    M = len(headers)
    NT = [1, 10, 25, 50, 75, 100]
    NF_DF = [int(M / 4), int(M / 2), int(3 * M / 4), "uniform"]
    F_DF = ['M/4', 'M/2', '3*M/4', 'Runif(1,M)']
    NF_RF = [1, 3, int(math.log2(M) + 1), int(math.sqrt(M))]
    F_RF = ['1', '3', 'log2(M+1)', 'sqrt(M)']
    outputdir = '../Output'
    df_results = pd.DataFrame(
        columns=['NT', 'F', 'NF', 'Accuracy'] + [str(x) for x in list(range(1, len(headers) + 1))])
    if args.algorithm == 'df':
        for number_trees in NT:
            i = 0
            print('NT={}'.format(number_trees))
            for number_features in NF_DF:
                print('    F={}'.format(F_DF[i]))
                decision_forest = DecisionForest(number_features=number_features, number_trees=number_trees)
                forest_results = decision_forest.make_and_test_forest(train_data, test_data)
                forest_results['F'] = F_DF[i]
                for _id in range(len(forest_results['FeatureImportance'])):
                    forest_results[str(_id + 1)] = '{}: {}'.format(forest_results['FeatureImportance'][_id][0],
                                                                   forest_results['FeatureImportance'][_id][1])
                del forest_results['FeatureImportance']
                df_results = df_results.append(forest_results, ignore_index=True)
                i += 1

    elif args.algorithm == 'rf':
        for number_trees in NT:
            i = 0
            print('NT={}'.format(number_trees))
            for number_features in NF_RF:
                print('    F={}'.format(F_RF[i]))
                random_forest = RandomForest(number_features=number_features, number_trees=number_trees)
                forest_results = random_forest.make_and_test_forest(train_data, test_data)
                forest_results['F'] = F_RF[i]
                for _id in range(len(forest_results['FeatureImportance'])):
                    forest_results[str(_id + 1)] = '{}: {}'.format(forest_results['FeatureImportance'][_id][0],
                                                                   forest_results['FeatureImportance'][_id][1])
                del forest_results['FeatureImportance']
                df_results = df_results.append(forest_results, ignore_index=True)
                i += 1

    outputfilename = outputdir + "{}_{}.csv".format(file_path[7:-5], args.algorithm)
    df_results.to_csv(outputfilename)
    print(df_results)