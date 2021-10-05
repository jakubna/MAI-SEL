import numpy as np
import pandas as pd
from more_itertools import set_partitions
import random


class DecisionForest:
    def __init__(self, number_features=2, number_trees=2):
        """
        :param k: Number of Clusters
        :param max_it: Maximum number of iterations if hasn't reached convergence yet.
        """
        self.data_column_names = None
        self.feature_indices = None
        self.forest = []
        self.predictions = []
        self.feature_importance = []
        self.number_trees = number_trees
        self.number_features = number_features
        self.num_cols = None
        self.type_cols = None
        self.accuracy = None

    def make_and_test_forest(self, train_data, test_data):
        the_forest = self.plant_forest(train_data)
        accuracy, self.predictions, self.multiple_predictions = self.make_prediction_forest(the_forest, test_data)
        self.accuracy = accuracy
        if sum(self.feature_importance) != 0:
            self.feature_importance = [x / sum(self.feature_importance) for x in self.feature_importance]
        else:
            self.feature_importance = [0] * len(self.feature_indices)
        results = {'Accuracy': round(accuracy,3), 'NF': self.number_features, 'NT': self.number_trees}
        feature_importance_values = []
        feature_importance_names = []
        for i in range(len(self.feature_importance)):
            feature_importance_values.append(round(self.feature_importance[i],3))
            feature_importance_names.append(self.data_column_names[i])
        zipped_importance = zip(feature_importance_names, feature_importance_values)
        sorted_importance = sorted(zipped_importance, key=lambda x: x[1], reverse=True)
        results['FeatureImportance'] = sorted_importance
        self.feature_importance = sorted_importance
        return results

    def plant_forest(self, train_data):
        self.forest = []
        self.data_column_names = list(train_data.columns)
        self.data_column_names.remove('class')
        train_dataset = train_data.values
        num_rows, num_cols = np.shape(train_dataset)
        self.num_cols = num_cols - 1
        class_index = self.num_cols
        self.feature_indices = list(range(class_index)) + list(range(class_index + 1, self.num_cols))
        self.type_cols = list(train_data.dtypes)
        self.feature_importance = [0] * len(self.feature_indices)

        for i in range(self.number_trees):
            if self.number_features == 'uniform':
                number_features = int(np.random.uniform(1, self.num_cols))
                random_features = random.sample(self.feature_indices, number_features)
            else:
                random_features = random.sample(self.feature_indices, self.number_features)
            the_tree = self.plant_tree(train_data, random_features)
            self.forest.append(the_tree)

        return self.forest

    def plant_tree(self, train_data, random_features):
        ''' start the process of recursion on the training data and let the tree
        grow to its max depth using subset of random features'''

        # get column names minus class
        # choose random set of features from column names

        root_node = self.find_best_split_point(train_data, random_features)
        self.recursive_splitter(root_node, random_features)
        return root_node

    def build_split(self, data, column_to_split, split_values):
        '''build 2 groups of data by splitting data on the column_to_split
           at the split_value'''
        left_split = data.loc[data[self.data_column_names[column_to_split]].isin(split_values[0])]
        right_split = data.loc[data[self.data_column_names[column_to_split]].isin(split_values[1])]

        return left_split, right_split

    def build_split_numeric(self, data, column_to_split, split_value):
        '''build 2 groups of data by splitting data on the column_to_split
           at the split_value'''
        left_split = data[data[column_to_split] < split_value]
        right_split = data[data[column_to_split] >= split_value]

        return left_split, right_split

    def multi_gini_index(self, group1, group2):
        '''Calculate Gini Impurity, func expects to be passed
           the 2 groups of data that are the result of a split'''
        class_proportions_group1 = group1['class'].value_counts(normalize=True)
        class_proportions_group2 = group2['class'].value_counts(normalize=True)

        instance_proportion_group1 = len(group1) / (len(group1) + len(group2))
        instance_proportion_group2 = len(group2) / (len(group1) + len(group2))

        gini1 = (1 - class_proportions_group1.pow(2).sum()) * (instance_proportion_group1)
        gini2 = (1 - class_proportions_group2.pow(2).sum()) * (instance_proportion_group2)
        gini = gini1 + gini2

        return gini

    def single_gini_index(self, group):
        '''Calculate Gini Impurity of a single group'''
        class_proportions = group['class'].value_counts(normalize=True)

        gini = (1 - class_proportions.pow(2).sum())

        return gini

    def find_best_split_point(self, passed_data, feature_subset):
        '''find best split point iterating over range of values returned from the
            passed data and return a dictionary which functions as a node '''
        best_split_gini = 10
        attribute_index = None
        best_split_value = None
        best_split_groups = None
        best_split_column = None
        best_split_type = None

        gini_X = self.single_gini_index(passed_data)
        for attribute_index in feature_subset:
            if self.type_cols[attribute_index] == 'O':
                attribute_values = list(set([x[attribute_index] for x in passed_data.values]))
                if len(attribute_values) == 1:
                    gini_XA = self.single_gini_index(passed_data)
                    if gini_XA < best_split_gini:
                        best_split_gini = gini_XA
                        best_split_column = attribute_index
                        best_split_value = attribute_values
                        best_split_groups = passed_data, pd.DataFrame(columns=passed_data.columns)
                        best_split_type = 0
                else:
                    partitions = list(set_partitions(attribute_values, 2))
                    for part in partitions:
                        if len(part[1]) < len(part[0]):
                            part = [part[1], part[0]]
                        left_split, right_split = self.build_split(passed_data, attribute_index, part)
                        gini_XA = self.multi_gini_index(left_split, right_split)
                        if gini_XA < best_split_gini:
                            best_split_gini = gini_XA
                            best_split_column = attribute_index
                            best_split_value = part
                            best_split_groups = left_split, right_split
                            best_split_type = 0
            else:
                col_name = passed_data.columns[attribute_index]
                split_point = float(passed_data[col_name].median())
                left_split, right_split = self.build_split_numeric(passed_data, col_name, split_point)
                gini_XA = self.multi_gini_index(left_split, right_split)

                if gini_XA < best_split_gini:
                    best_split_gini = gini_XA
                    best_split_column = attribute_index
                    best_split_value = split_point
                    best_split_groups = left_split, right_split
                    best_split_type = 1

        gini_A = gini_X - best_split_gini
        self.feature_importance[best_split_column] += gini_A
        return {'column_id': best_split_column, 'column_name': self.data_column_names[best_split_column],
                'type': best_split_type, 'dsplit_value': best_split_value,
                'gini': best_split_gini, 'groups': best_split_groups}

    def recursive_splitter(self, node, random_features):
        '''this function recursively splits the data starting with the root node which its passed
        until the groups are homogenous or further splits result in empty nodes'''
        left_group, right_group = node['groups']
        # delete the groups entry in original node
        del node['groups']
        # check if the groups of the node are empty
        if left_group.empty or right_group.empty:
            # combine as we will use original to predict
            combined = pd.concat([left_group, right_group])
            predicted_class = combined['class'].value_counts().index[0]
            node['left'] = node['right'] = predicted_class
            return [predicted_class]
        # check if the groups of the node are homogenous otherwise call recursive_spltter again
        if self.single_gini_index(left_group) == 0:
            predicted_class = left_group['class'].value_counts().index[0]
            node['left'] = predicted_class
        else:
            node['left'] = self.find_best_split_point(left_group, random_features)
            curr_node = self.recursive_splitter(node['left'], random_features)
            if type(curr_node) == list:
                node['left'] = curr_node[0]

        if self.single_gini_index(right_group) == 0:
            predicted_class = right_group['class'].value_counts().index[0]
            node['right'] = predicted_class
        else:
            node['right'] = self.find_best_split_point(right_group, random_features)
            curr_node = self.recursive_splitter(node['right'], random_features)
            if type(curr_node) == list:
                node['right'] = curr_node[0]
        return node

    def make_prediction_tree(self, data_row, root_node):
        '''recursively traverse the tree from root to leaf turning left if feature value
        to test is less than dsplit_value or right otherwise until we reach a leaf node'''

        if root_node['type'] == 0:
            # check if feature of data_row is less than dsplit_value else move to right branch
            if data_row[root_node['column_id']] in root_node['dsplit_value'][0]:
                # check if at a branch or a leaf if branch recursively call predict else return leaf prediction
                if type(root_node['left']) is dict:
                    return self.make_prediction_tree(data_row, root_node['left'])
                else:
                    return root_node['left']
            else:
                if type(root_node['right']) is dict:
                    return self.make_prediction_tree(data_row, root_node['right'])
                else:
                    return root_node['right']
        else:
            if data_row[root_node['column_id']] < root_node['dsplit_value']:
                # check if at a branch or a leaf if branch recursively call predict else return leaf prediction
                if type(root_node['left']) is dict:
                    return self.make_prediction_tree(data_row, root_node['left'])
                else:
                    return root_node['left']
            else:
                if type(root_node['right']) is dict:
                    return self.make_prediction_tree(data_row, root_node['right'])
                else:
                    return root_node['right']

    def make_prediction_forest(self, forest, test_data):
        classes = test_data['class']
        classes = classes.reset_index(drop=True)

        forest_predictions = []
        multiple_forest_predictions = []
        for index, row in test_data.iterrows():
            tree_predictions = []
            for tree in forest:
                tree_predictions.append(self.make_prediction_tree(row, tree))
            multiple_forest_predictions.append(tree_predictions)
            tree_predictions_series = pd.Series(tree_predictions)
            predicted_class = tree_predictions_series.value_counts().index[0]
            forest_predictions.append(predicted_class)
        forest_pred_series = pd.Series(forest_predictions)
        results = forest_pred_series == classes
        successes = 0
        for i in results:
            if i == True: successes += 1
        accuracy = successes / len(classes)
        return accuracy, forest_pred_series, multiple_forest_predictions
