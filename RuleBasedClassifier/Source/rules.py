import numpy as np
import pandas as pd
import itertools
import datetime


class RULES:
    def __init__(self):
        """
        RULES classifier.
        """
        self.number_attributes = None
        self.number_combinations = 1
        self.rules = []
        self.X = None
        self.y = None
        self.feature_indices = None
        self.headers = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Specify rules.
        :param X: 2D data array of size (rows, features).
        :param y: 1D vector with all y attributes.
        """
        ### VARIABLES
        self.headers = list(X.columns)
        self.number_attributes = len(self.headers)
        self.feature_indices = list(range(self.number_attributes))
        self.X = X.values
        self.X = np.array(X)
        self.y = y
        self.y = np.array(y)
        self.unclassified = self.X.copy()
        self.rules_number = 0

        start_time = datetime.datetime.now()
        # Continue as long as there are unclassified examples.
        while self.unclassified.shape[0] > 0 and self.number_combinations <= self.number_attributes:
            iteration_rules = {}
            for feature_combination in itertools.combinations(self.feature_indices, self.number_combinations):
                # Find all selectors(pairs Attribute-Value) from NON-classified instances
                selectors = []
                for combination in range(self.number_combinations):
                    unique_values = np.unique(self.unclassified[:, feature_combination[combination]])
                    selectors.append(unique_values)
                # Form conditions as a combination of NumberCombinations selectors
                conditions = list(itertools.product(*selectors))
                for condition in conditions:
                    indices = []
                    # Check if there are instances which satisfies the condition
                    for index in range(self.X.shape[0]):
                        if_conditions_satisfied = True
                        for n in range(self.number_combinations):
                            if self.X[index][feature_combination[n]] != condition[n]:
                                if_conditions_satisfied = False
                        if if_conditions_satisfied:
                            indices.append(index)
                    classes = self.y[indices]

                    # Check if all the instances that satisfied condition are from the same class
                    if len(np.unique(classes)) == 1:
                        # Check if there are unclassified instances which satisfy condition
                        satisfy_indices = []
                        for instance_x in self.X[indices]:
                            for i, instance_unclassified in enumerate(self.unclassified):
                                if np.array_equal(instance_unclassified, instance_x):
                                    if i not in satisfy_indices:
                                        satisfy_indices.append(i)
                        if len(satisfy_indices) > 0:
                            irrelevant_rule = False
                            assigned_class = np.unique(classes)[0]
                            # check the relevance of the condition against previously obtained rules
                            # based on combinations obtained for a current iteration
                            try:
                                irrelevant_rule = self.__check_rule(iteration_rules[assigned_class], indices)
                            except KeyError:
                                pass
                            if irrelevant_rule == False:
                                key = (feature_combination, condition)
                                try:
                                    iteration_rules[assigned_class][key] = indices
                                except:
                                    iteration_rules[assigned_class] = {key: indices}
                                coverage_instances = len(satisfy_indices)
                            else:
                                # Delete the rule recognized as irrelevant
                                cov_del_rule = 0
                                for r in range(len(self.rules)):
                                    if self.rules[r][0] == irrelevant_rule[0][0]:
                                        if self.rules[r][1] == irrelevant_rule[0][1]:
                                            cov_del_rule = self.rules[r][3]
                                            self.rules.pop(r)
                                            break
                                coverage_instances = len(satisfy_indices) + cov_del_rule
                            # Create new rule
                            new_rule = (feature_combination, condition, assigned_class, coverage_instances, indices)
                            self.rules.append(new_rule)
                            # Delete all the examples that are covered by the new rule
                            self.unclassified = np.delete(self.unclassified, satisfy_indices, 0)
            self.number_combinations += 1

        end_time = datetime.datetime.now()
        cov_sum = 0
        for r in self.rules:
            cov_sum += r[3]
            self.__print_rule(r[0], r[1], r[2])
            print('Coverage: {} instances {:.2f}% of all instances'.
                  format(r[3], 100 * r[3] / self.X.shape[0]))
            print('Precision: {:.2f}%'.format(self.__count_precision(r[4], r[2]) * 100))
        print('Number of derived rules: ' + str(len(self.rules)))
        print("Training time: {}".format(end_time - start_time))
        print("Covered instances: {}".format(cov_sum))

    def predict(self, X: np.ndarray):
        """
        Assign labels to a list of observations.
        :param x: 2D data array of size (rows, features).
        :return: Labels assigned to each row of X.
        """
        if len(self.rules) == 0:
            raise Exception('Fit the model with some data before running a prediction')
        y_pred = []
        i = 0
        for instance in X.values:
            i += 1

            for rule_index, rule in enumerate(self.rules):
                if_condition_satisfied = True
                for (x, y) in zip(rule[0], range(len(rule[0]))):
                    if instance[x] != rule[1][y]:
                        if_condition_satisfied = False
                        break
                if if_condition_satisfied:
                    y_pred.append(rule[2])
                    break
            if i != len(y_pred):
                y_pred.append('zero')
        return y_pred

    def __check_rule(self, prev_rules, indices):
        """
        Check irrelevant conditions against previously obtained rules.
        :param prev_rules: dict of previously obtained rules
        :param indices: the indices of rows that potential candidat rule classifies
        :return: information about an irrelevant rule if exists
        """
        irrelevant_rule = False
        for key, value in prev_rules.items():
            if set(value).issubset(set(indices)):
                if len(indices) > len(value):
                    irrelevant_rule = (key, value)
                    break
        return irrelevant_rule

    def __print_rule(self, feature_combination, attribute_values, assigned_class):
        """
        Display rules, coverage and precission in an interpretable way.
        :param feature_combination: indices of combination's conditions
        :param attribute_values: values of attrbute_values
        :param assigned_class: label
        """
        self.rules_number += 1
        out = 'R' + str(self.rules_number) + ': IF '
        for feature_index, attribute_value in zip(feature_combination, attribute_values):
            out += self.headers[feature_index] + ' = ' + str(attribute_value) + ' AND '
        print(out[:-4] + 'THEN ' + str(assigned_class))

    def __count_precision(self, indices, assigned_class):
        """
        Compue prevision of the rule.
        :param assigned_class: (str) label
        :param indices: (list) indices of instances covered by the rule
        :return: information about an irrelevant rule if exists
        """
        return (np.array(self.y)[indices] == assigned_class).sum() / len(np.array(self.y)[indices])