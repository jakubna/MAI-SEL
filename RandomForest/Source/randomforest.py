from decisionforest import DecisionForest
import numpy as np
import pandas as pd
import random
from sklearn.utils import resample


class RandomForest(DecisionForest):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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

        # resample training data for bagging
        resampled_training_sets = []
        for i in range(self.number_trees):
            dataset = resample(train_data, replace=True)
            # dataset = train_data.sample(frac=fraction_samples,replace=True)
            resampled_training_sets.append(dataset)
        for dataset in resampled_training_sets:
            the_tree = self.plant_tree(train_data)
            self.forest.append(the_tree)

        return self.forest

    def plant_tree(self, train_data):
        ''' start the process of recursion on the training data and let the tree
        grow to its max depth using subset of random features'''

        # get column names minus class
        # choose random set of features from column names
        random_features = random.sample(self.feature_indices, self.number_features)
        root_node = self.find_best_split_point(train_data, random_features)
        self.recursive_splitter(root_node)
        return root_node

    def recursive_splitter(self, node):
        '''this function recursively splits the data starting with the root node which its passed
        untill the groups are homogenous or further splits result in empty nodes'''
        random_features = random.sample(self.feature_indices, self.number_features)
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
            curr_node = self.recursive_splitter(node['left'])
            if type(curr_node) == list:
                node['left'] = curr_node[0]

        if self.single_gini_index(right_group) == 0:
            predicted_class = right_group['class'].value_counts().index[0]
            node['right'] = predicted_class
        else:
            node['right'] = self.find_best_split_point(right_group, random_features)
            curr_node = self.recursive_splitter(node['right'])
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
