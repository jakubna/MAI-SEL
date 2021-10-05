from rules import RULES
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import argparse
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def data_preprocessing(X_set, y_set, shuffle=True, test_size=0.2, seed=11, mf=False):
    """
    Apply the personalized operations to preprocess the database.
    :param X: 2D data array of size (rows, features).
    :param y: 1D vector with all y attributes.
    :param shuffle: Whether or not to shuffle the data before splitting. Default = True
    :param test_size: Float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split. Default: 0.2
    :param seed: Seed with which the random number generator is initialized. Default: 42
    :param mf: If dataset contains missing values. Default: False
    :return:
            splitting: List containing train-test split of inputs.
    """
    if mf:
        # detect missing values and replacing using the most frequent value
        meta = list(X_set.columns)
        if X_set.isnull().any().sum() > 0:
            for x in meta:
                top_frequent_value = X_set[x].describe()['top']
                X_set[x].fillna(top_frequent_value, inplace=True)

    splitting = train_test_split(X_set, y_set, test_size=test_size, random_state=seed, shuffle=shuffle)
    return splitting

parser = argparse.ArgumentParser('main')
parser.add_argument('dataset_size',
                    help='Size of the dataset. Possible values: [small, medium, large]', type=str)
args = parser.parse_args()

if args.dataset_size not in ['small', 'medium', 'large']:
    raise Exception("Wrong dataset size. Possible values: [small, medium, large]")

if args.dataset_size == "small":
    file_path = './Data/balance-scale.data'
    df = pd.read_csv(file_path, sep=',')
    X_set = df.iloc[:, 1:]
    y_set = df.iloc[:, 0]
    x_train, x_test, y_train, y_test = data_preprocessing(X_set, y_set)
    db_name = "balance-scale"
elif args.dataset_size == "medium":
    file_path = './Data/car.data'
    df = pd.read_csv(file_path, sep=',')
    X_set = df.iloc[:, :-1]
    y_set = df.iloc[:, -1]
    x_train, x_test, y_train, y_test = data_preprocessing(X_set, y_set)
    db_name = "car"
elif args.dataset_size == "large":
    file_path = './Data/nursery.data'
    df = pd.read_csv(file_path, sep=',')
    X_set = df.iloc[:, :-1]
    y_set = df.iloc[:, -1]
    x_train, x_test, y_train, y_test = data_preprocessing(X_set, y_set)
    db_name = "nursery"

print("Name of dataset: {}".format(db_name))
print("Number of training instances: {}".format(x_train.shape[0]))
print("Number of test instances: {}".format(x_test.shape[0]))

rules = RULES()
rules.fit(x_train, y_train)
y_pred = rules.predict(x_test)
print("Accuracy: {}".format(accuracy_score(y_test, y_pred)))