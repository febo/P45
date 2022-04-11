# P45 - A Python C4.5 implementation
#
# Copyright 2022 Fernando Esteban Barril Otero
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import math
import numpy as np
import pandas as pd
import random

from collections import Counter
from datetime import datetime
from json import load
from pandas.api.types import is_numeric_dtype

# minimum difference between continuous values
DELTA = 1e-5

# minumum number of instances for each node
MINIMUM_INSTANCES = 2

# correction value used in the threshold and error calculation (based on C4.5)
EPSILON = 1e-3

# values below this are considered zero for gain ratio calculation
ZERO = 1e-7

# make sure it is not trying to modify a copy of a dataframe
pd.options.mode.chained_assignment = 'raise'

# sets a fixed random seed
random.seed(1000)

# (dis/en)able the pruning
PRUNE = True

#
# Gain Ratio calculation
# ---------


def gain_ratio(data, attribute, weights):
    """Calculates the gain ratio of the specified attribute

    Parameters
    ----------
    data : DataFrame
        The current data
    attribute : str
        The name of the attribute
    weights : np.array
        The instance weights to be used in the length calculation

    Returns
    -------
    tuple
        a tuple representing the (gain ratio, gain, split information) of the attribute

    """

    # class values present in the data
    filtered = data[attribute].notna()
    class_values = list(data.iloc[data[filtered].index, -1].unique())
    S = []

    for c in class_values:
        membership = (data.iloc[data[filtered].index, -1] == c)
        S.append(weights[membership & filtered].sum())

    values = list(filter(lambda v: v == v, data[attribute].unique()))
    #  number of missing values
    missing = weights[data[attribute].isna()].sum()
    # sum of instances with known values
    length = weights.sum() - missing
    total_entropy = 0

    # calculates the entropy of the whole data

    for s in S:
        p = s / length
        total_entropy -= (p * np.log2(p))

    # calculates the entropy of the partition based on the attribute

    partition_entropy = 0
    partition_split = 0

    for v in values:
        partition = (data[attribute] == v)
        partition_length = weights[partition].sum()
        entropy = 0

        for c in class_values:
            membership = (data.iloc[data[partition].index, -1] == c)
            count = weights[membership & partition].sum()

            if count > 0:
                p = count / partition_length
                entropy -= (p * np.log2(p))

        partition_entropy += (partition_length / length) * entropy

        split = partition_length / (length + missing)
        partition_split -= split * np.log2(split)

    if missing > 0:
        m = missing / (length + missing)
        partition_split -= m * np.log2(m)

    gain = (length / (length + missing)) * (total_entropy - partition_entropy)

    return gain / partition_split, gain, partition_split


def candidate_thresholds(data, attribute):
    """Generates the candidates threshold values for a continuous attribute

    Parameters
    ----------
    data : DataFrame
        The current data
    attribute : str
        The name of the attribute

    Returns
    -------
    list
        a list of values representing the candidate threshold values

    """

    sorted_attribute = np.array(
        [v for v in data[attribute] if not np.isnan(v)])
    sorted_attribute.sort()

    thresholds = []
    count = 0

    for s in range(len(sorted_attribute) - 1):
        count += 1
        if (sorted_attribute[s] + DELTA) < sorted_attribute[s + 1] and count >= 2 and (len(sorted_attribute) - count) >= 2:
            thresholds.append(
                (sorted_attribute[s] + sorted_attribute[s+1]) / 2)

    return thresholds


def gain_ratio_numeric(data, attribute, weights):
    """Calculates the gain ratio of the specified numeric attribute

    This function evaluates multiple thresholds values to dynamically discretise the
    continuous attribute and calculate the gain ratio information.

    Parameters
    ----------
    data : DataFrame
        The current data
    attribute : str
        The name of the attribute
    weights : np.array
        The instance weights to be used in the length calculation

    Returns
    -------
    tuple
        a tuple representing the (gain ratio, gain, threshold) of the attribute

    """

    thresholds = candidate_thresholds(
        data, attribute)  # the list of threshold values

    if len(thresholds) == 0:
        return 0, 0, 0

    valid_data = data[data[attribute].notna()].copy().reset_index(drop=True)
    valid_weights = weights[data[attribute].notna()]
    # saves a copy of the original values
    values = valid_data[attribute].copy()

    #  number of missing values
    missing = weights[data[attribute].isna()].sum()
    # sum of instances with known outcome
    length = valid_weights.sum()

    gain_correction = length / weights.sum()
    split_correction = missing / weights.sum()
    penalty = np.log2(len(thresholds)) / weights.sum()
    gain_information = []

    for t in thresholds:
        # create a binary column representing the threshold division
        binary_split = ['H' if v > t else 'L' for v in values]
        valid_data[attribute] = binary_split

        _, gain, split = gain_ratio(valid_data, attribute, valid_weights)

        # apply a penalty for evaluating multiple threshold values (based on C4.5)
        if missing > 0:
            gain = (gain_correction * gain) - penalty
            split = (split_correction * split)
        else:
            gain -= penalty

        gain_information.append((gain / split, gain))

    average_gain = sum([g[1] for g in gain_information]) / len(thresholds)
    valid_thresholds = []

    for candidate in gain_information:
        # candidate gain needs to be higher than the average gain
        if candidate[1] > (average_gain - EPSILON):
            valid_thresholds.append(candidate[0])
        else:
            valid_thresholds.append(0)

    selected = np.argmax(valid_thresholds)

    return gain_information[selected][0], gain_information[selected][1], thresholds[selected]

#
# Attribute selection
# ---------


def search_best_attribute(data, weights):
    """Search for the best attribute to create a decision node

    This function searches for the best attribute for a decision node. For each attribute,
    it calculates the gain ratio. The selected attribute is the one with the highest
    gain ratio.

    Parameters
    ----------
    data : DataFrame
        The current data
    weights : np.array
        The instance weights to be used in the length calculation

    Returns
    -------
    str, tuple
        the name of the selected attribute and its associated information. For categorical
        attributes, it is a tuple (gain ratio, gain, split information); for continuous
        attributes, it is a tuple (gain ratio, gain, threshold)
    """

    predictors = data.iloc[:, 0:-1]

    if len(predictors.columns) == 0:
        # no attributes left to choose
        return None, (0, 0, 0)

    candidates = []
    average_gain = 0

    for attribute in predictors.columns:
        if is_numeric_dtype(data[attribute]):
            c = attribute, gain_ratio_numeric(data, attribute, weights)
        else:
            c = attribute, gain_ratio(data, attribute, weights)

        average_gain += c[1][1]
        candidates.append(c)

    average_gain = (average_gain / len(candidates)) - EPSILON
    selected = None
    # [0] gain ratio
    # [1] gain
    # [2] split informaton / threshold
    best = (ZERO, ZERO, ZERO)

    for attribute, c in candidates:
        if c[0] > best[0] and c[1] >= average_gain:
            selected = attribute
            best = c

    return selected, best

#
# Error estimation
# ---------


coefficient_value = [0, 0.001, 0.005, 0.01, 0.05, 0.10, 0.20, 0.40, 1.00]
deviation = [4.0, 3.09, 2.58, 2.33, 1.65, 1.28, 0.84, 0.25, 0.00]
CF = 0.25

coefficient_index = 0

while CF > coefficient_value[coefficient_index]:
    coefficient_index += 1

coefficient = deviation[coefficient_index - coefficient_index] + (deviation[coefficient_index] - deviation[coefficient_index - 1]) * (
    CF - coefficient_value[coefficient_index - 1]) / (coefficient_value[coefficient_index] - coefficient_value[coefficient_index - 1])
coefficient = coefficient * coefficient


def estimate_error(total, errors):
    """Estimates the prediction error.

    Parameters
    ----------
    total : int
        The total number of predictions
    errors : int
        The number of incorrect predictions

    Returns
    -------
    int
        the estimated errors

    """

    if errors < 1e-6:
        return total * (1 - math.exp(math.log(CF) / total))
    elif errors < 0.9999:
        v = total * (1 - math.exp(math.log(CF) / total))
        return v + errors * (estimate_error(total, 1.0) - v)
    elif errors + 0.5 >= total:
        return 0.67 * (total - errors)
    else:
        pr = (errors + 0.5 + coefficient / 2 + math.sqrt(coefficient * ((errors + 0.5)
              * (1 - (errors + 0.5) / total) + coefficient / 4))) / (total + coefficient)
        return (total * pr - errors)

#
# Data structures
# ---------


class Operator:
    """
    Enum-like class to represent different operators
    """

    EQUAL = 1
    LESS_OR_EQUAL = 2
    GREATER = 3


class Node:
    """
    A class used to represent a node of the decision tree.

    Each node can have a number of child nodes (internal nodes) or none (leaf nodes).
    The root of the tree is also represented as a node.
    """

    def __init__(self, attribute, parent=None, error=0, total=0, distribution=None):
        """
        Parameters
        ----------
        attribute : str
            The name of the attribute represented by the node (internal nodes) or
            the class value predicted (leaf nodes)
        parent : Node, optional
            The parent node of the node
        error:
            The number of prediction errors (leaf nodes)
        total:
            The number of instances reaching the node (leaf nodes)
        """

        self.attribute = attribute
        self.parent = parent

        # private-like attributes
        self._error = error
        self._total = total
        self._distribution = distribution
        self._classes = []

        self.level = 0 if parent is None else parent.level + 1

        self.children = []
        self.conditions = []
        self.operators = []

    @property
    def classes(self):
        """Return the list of classes that the node (tree) can predict.

        This method can only be used on the root node of the tree.

        Returns
        -------
        list
            the list of classes that the node (tree) can predict.
        """

        return self._classes

    def add(self, node, condition, operator=Operator.EQUAL):
        """Adds a child node

        The node will be added at the end of a branch. The condition and operator are
        used to decide when to follow the branch to the node

        Parameters
        ----------
        node : Node
            The node to add
        condition : str or float
            The value to decide when to choose to visit the node
        operator : Operator, optional
            The operator to decide when to choose to visit the node
        """

        node.parent = self
        self.children.append(node)
        self.conditions.append(condition)
        self.operators.append(operator)

    def to_text(self):
        """Prints a textual representation of the node

        This method prints the node and any of its child nodes
        """

        self.__print_node("")

    def __print_node(self, prefix):
        """Prints a textual representation of the node

        This method prints the node and any of its child nodes recusively

        Parameters
        ----------
        prefix : str
            The prefix to be used to print the node
        """

        if self.is_leaf():
            if self._error > 0:
                print("{} ({:.1f}/{:.1f})".format(self.attribute,
                                                  self._total,
                                                  self._error), end="")
            else:
                print("{} ({:.1f})".format(self.attribute,
                                           self._total), end="")
        else:
            if len(prefix) > 0:
                print("")

            for i, v in enumerate(self.conditions):
                if i > 0:
                    print("")

                operator = None

                if self.operators[i] == Operator.EQUAL:
                    operator = "="
                elif self.operators[i] == Operator.LESS_OR_EQUAL:
                    operator = "<="
                elif self.operators[i] == Operator.GREATER:
                    operator = ">"

                print("{}{} {} {}: ".format(
                    prefix, self.attribute, operator, v), end="")

                self.children[i].__print_node(prefix + "|    ")

    def is_leaf(self):
        """Checks whether the node is a leaf node

        Returns
        -------
        bool
            True if the node is a leaf node; otherwise False
        """

        return len(self.conditions) == 0

    def predict(self, instance):
        """Classify the specified instance

        This method expects that the instance (row) if a slice of a dataframe with the
        same attributes names as the one used to create the tree

        Parameters
        ----------
        instance : DataFrame slice
            The instance (row) to be classified

        Returns
        -------
        str
            The class value predicted

        """

        probabilities = Node.__predict(instance, self, 1.0)
        prediction = ("", 0)

        for value, count in probabilities.items():
            if count > prediction[1]:
                prediction = (value, count)

        if prediction[1] > 0:
            return prediction[0]

        raise Exception("Could not predict a value")

    def probabilities(self, instance):
        """Classify the specified instance, returning the probability of each class value
        prediction.

        This method expects that the instance (row) is a slice of a dataframe with the
        same attributes names as the one used to create the tree. The order of the class
        values is determined by the ``self.classes`` property.

        Parameters
        ----------
        instance : DataFrame slice
            The instance (row) to be classified

        Returns
        -------
        list
            list of class value probabilities

        """

        probabilities = Node.__predict(instance, self, 1.0)
        prediction = []

        for value in self.classes:
            prediction.append(probabilities[value])

        return prediction

    def __predict(instance, node, weight):
        """Classify the specified instance

        This method expects that the instance (row) if a slice of a dataframe with the
        same attributes names as the one used to create the tree

        Parameters
        ----------
        instance : DataFrame slice
            The instance (row) to be classified

        Returns
        -------
        str
            The class value predicted

        """

        probabilities = Counter()

        # in case the node is a leaf
        if node.is_leaf():
            for value, count in node._distribution.items():
                probabilities[value] = weight * (count / node._total)

            return probabilities

        # if not, find the branch to follow
        value = instance[node.attribute]

        if pd.isna(value):
            total = node.total()

            for i, v in enumerate(node.conditions):
                w = node.children[i].total() / total
                probabilities += Node.__predict(instance,
                                                node.children[i], weight * w)
        else:
            match = False

            for i, v in enumerate(node.conditions):
                if node.operators[i] == Operator.EQUAL and value == v:
                    match = True
                elif node.operators[i] == Operator.LESS_OR_EQUAL and value <= v:
                    match = True
                elif node.operators[i] == Operator.GREATER and value > v:
                    match = True

                if match:
                    probabilities += Node.__predict(instance,
                                                    node.children[i], weight)
                    break

            if not match:
                raise Exception(
                    f"Cound not match value {value} for attribute {node.attribute}")

        return probabilities

    def total(self):
        """Returns the number of instances reaching the node

        For internal nodes, this is the sum of the total from its child nodes

        Returns
        -------
        int
            the number of instances reaching the node

        """

        if self.is_leaf():
            return self._total
        else:
            t = 0
            for node in self.children:
                t += node.total()
            return t

    def error(self):
        """Returns the number of prediction errors observed during the creation of the tree

        For internal nodes, this is the sum of the errors from its child nodes

        Returns
        -------
        int
            the number of prediction errors observed during the creation of the tree

        """

        if self.is_leaf():
            return self._error
        else:
            e = 0
            for node in self.children:
                e += node.error()
            return e

    def estimated(self):
        """Returns the number of estimated errors observed during the creation of the tree

        For internal nodes, this is the sum of the estimated errors from its child nodes

        Returns
        -------
        float
            the number of estimated errors observed during the creation of the tree

        """

        if self.is_leaf():
            return self._error + estimate_error(self._total, self._error)
        else:
            e = 0
            for node in self.children:
                e += node.estimated()
            return e

    def adjust(self, data):
        """Replaces the threshold values of continuous attributes with values that occur
        on the training data

        The discretisation uses the average value between two consecutive values to
        evaluate thresholds.

        Parameters
        ----------
        data : DataFrame
            The training data
        """

        if not self.is_leaf():
            ordered = []
            # we only need to look at one of the operators/conditions since the
            # threshold value will be the same in both branches
            operator = self.operators[0]
            threshold = self.conditions[0]

            if operator == Operator.LESS_OR_EQUAL or operator == Operator.GREATER:
                sorted_values = np.array(data[self.attribute])
                sorted_values.sort()
                selected = threshold

                for v in sorted_values:
                    if v > threshold:
                        break
                    selected = v

                self.conditions = [selected] * len(self.conditions)

            for child in self.children:
                child.adjust(data)

    def estimate_with_data(self, data, weights, update=False):
        """Returns the number of estimated errors observed on the specified data, updating
        the values if update=True (default False)

        For internal nodes, this is the sum of the estimated errors from its child nodes

        Parameters
        ----------
        data : DataFrame
            The data to use
        weights : np.array
            The instance weights to be used in the length calculation
        update : bool
            Indicate whether the error values should be updated or not

        Returns
        -------
        float
            the number of estimated errors
        """

        if self.is_leaf():
            class_attribute = data.iloc[:, -1]
            total = weights.sum()
            correct_predictions = 0 if total == 0 else weights[class_attribute == self.attribute].sum(
            )
            error = total - correct_predictions

            if update:
                # class value = count
                distribution = Counter()

                for value in class_attribute.unique():
                    distribution[value] = weights[class_attribute == value].sum()

                self._distribution = distribution
                self._total = total
                self._error = error

            return error + estimate_error(total, error)
        else:
            missing = data[self.attribute].isna()
            known_length = weights.sum() - weights[missing].sum()

            total = 0.0

            for i, v in enumerate(self.conditions):
                if self.operators[i] == Operator.EQUAL:
                    partition = (data[self.attribute] == v)
                elif self.operators[i] == Operator.LESS_OR_EQUAL:
                    partition = (data[self.attribute] <= v)
                elif self.operators[i] == Operator.GREATER:
                    partition = (data[self.attribute] > v)

                updated_weights = weights.copy()
                w = weights[partition].sum() / known_length

                updated_weights[missing] = updated_weights[missing] * w
                updated_weights = updated_weights[partition | missing]

                if is_numeric_dtype(data[self.attribute]):
                    branch_data = data[partition |
                                       missing].reset_index(drop=True)
                else:
                    branch_data = data[partition | missing].drop(
                        columns=self.attribute).reset_index(drop=True)

                total += self.children[i].estimate_with_data(
                    branch_data, updated_weights, update)

            return total

    def sort(self):
        """Sort the branches of the node, placing leaf nodes at the start of the children
        array.

        This method improves the shape of the node (tree) for visualisation - there is
        no difference it terms of the performance of the tree.
        """

        for i in range(len(self.children)):
            if not self.children[i].is_leaf():
                to_index = -1

                for j in range(1, len(self.children)):
                    if self.children[j].is_leaf():
                        to_index = j
                        break

                if to_index == -1:
                    self.children[i].sort()
                else:
                    child = self.children[to_index]
                    condition = self.conditions[to_index]
                    operator = self.operators[to_index]

                    for j in range(to_index, i, -1):
                        self.children[j] = self.children[j - 1]
                        self.conditions[j] = self.conditions[j - 1]
                        self.operators[j] = self.operators[j - 1]

                    self.children[i] = child
                    self.conditions[i] = condition
                    self.operators[i] = operator

#
# Building the tree
# ---------


def calculate_majority(class_attribute, weights):
    """Finds the majority class value based on the weights of the instances

    Note that in case there are more than one value with the same distribution,
    a random value among those is returned.

    Parameters
    ----------
    class_attribute : Series
        The class values of the instances
    weights : np.array
        The instance weights to be used in the majority calculation

    Returns
    -------
    str
        the majority class value
    """

    majority = []
    best = 0

    for value in class_attribute:
        count = weights[class_attribute == value].sum()

        if count > best:
            majority.clear()
            majority.append(value)
            best = count
        elif count == best:
            majority.append(value)

    return majority[random.randrange(len(majority))] if len(majority) > 0 else None


def pre_prune(data, node, weights):
    """Performs a pre-pruning test to decide whether to replace an internal node by a leaf
    node or not

    Parameters
    ----------
    data : DataFrame
        The training data
    Node : Node
        The node undergoing pruning
    weights : np.array
        The instance weights to be used in the length calculation

    Returns
    -------
    Node
        the node in case its error is lower; otherwise a leaf node to be used instead
    """

    class_attribute = data.iloc[:, -1]
    majority = calculate_majority(class_attribute, weights)

    length = weights.sum()
    correct_predictions = 0 if length == 0 else weights[class_attribute == majority].sum(
    )
    leaf_error = length - correct_predictions

    if node.error() >= leaf_error - EPSILON:
        # class value = count
        distribution = Counter()

        for value in class_attribute.unique():
            distribution[value] = weights[class_attribute == value].sum()

        return Node(majority, node.parent, leaf_error, length, distribution)

    return node


def post_prune(data, node, majority, weights):
    """Performs a pessimistic prunnig on the newly created subtree, which prunes
    nodes based on an estimated error penalty

    The pruning procedure involves:

    1. estimating the node (sub-tree) error
    2. estimating the error if the node (sub-tree) is replaced by a leaf node
    3. estimating the error if the node is replaced by the most frequent branch

    In case of (2) and (3) generating a smaller error, the node is replaced by
    the correspnding node.

    Parameters
    ----------
    data : DataFrame
        The training data
    Node : Node
        The node undergoing pruning
    majority : str
        The majority class value if the node is replaced by a leaf node
    weights : np.array
        The instance weights to be used in the length calculation

    Returns
    -------
    node
        the pruned node
    """

    # (1) subtree error

    subtree_error = node.estimated()

    # (2) leaf error

    class_attribute = data.iloc[:, -1]
    # class value = count
    distribution = Counter()
    leaf_total = 0

    for value in class_attribute.unique():
        distribution[value] = weights[class_attribute == value].sum()
        leaf_total += distribution[value]

    correct_predictions = 0 if leaf_total == 0 else distribution[majority]
    leaf_error = leaf_total - correct_predictions
    leaf_error += estimate_error(leaf_total, leaf_error)

    # (3) branch error

    selected = node.children[0]

    for i in range(1, len(node.children)):
        if selected.total() < node.children[i].total():
            selected = node.children[i]

    # checks whether to prune the node or not

    branch_error = float('inf')

    if selected.is_leaf():
        branch_error = leaf_error
    else:
        branch_error = selected.estimate_with_data(data, weights)

    if leaf_error <= (subtree_error + 0.1) and leaf_error <= (branch_error + 0.1):
        # replace by a leaf node
        return Node(majority, node.parent, leaf_error, leaf_total, distribution)
    elif branch_error <= (subtree_error + 0.1):
        # replace by the most common branch
        selected.estimate_with_data(data, weights, True)
        selected.parent = node.parent
        return selected

    return node


def build_decision_nodes(data, tree=None, parent_majority=None, weights=None):
    """Builds a decision tree

    Parameters
    ----------
    data : DataFrame
        The training data
    tree : Node
        The parent node
    parent_majority : str
        The majority class value of the parent
    weights : np.array
        The instance weights to be used in the length calculation

    Returns
    -------
    node
        the node representing the root of the (sub-)tree created
    """

    if weights is None:
        # initializes the weights of the instances
        weights = np.ones(len(data))

    class_attribute = data.iloc[:, -1]
    is_unique = len(class_attribute.unique()) == 1
    length = weights.sum()

    # majority class (mode can return more than one value)
    majority = parent_majority if length == 0 else calculate_majority(
        class_attribute, weights)

    # if all instance belong to the same class or there is no enough data
    # a leaf node is added

    if is_unique or length <= (MINIMUM_INSTANCES * 2):
        correct_predictions = 0 if length == 0 else weights[class_attribute == majority].sum(
        )
        # class value = count
        distribution = Counter()

        for value in class_attribute.unique():
            distribution[value] = weights[class_attribute == value].sum()

        return Node(majority, tree, length - correct_predictions, length, distribution)

    # search the best attribute for a split

    attribute, info = search_best_attribute(data, weights)

    if attribute == None:
        # adds a leaf node, could not select an attribute
        correct_predictions = 0 if length == 0 else weights[class_attribute == majority].sum(
        )
        # class value = count
        distribution = Counter()

        for value in class_attribute.unique():
            distribution[value] = weights[class_attribute == value].sum()

        return Node(majority, tree, length - correct_predictions, length, distribution)

    # adjusts the instance weights based on missing values
    missing = data[attribute].isna()
    known_length = length - weights[missing].sum()

    # (count, adjusted count)
    distribution = []
    # only present if the selected attribute is categorical
    attribute_values = []

    if is_numeric_dtype(data[attribute]):
        # lower partition
        count = weights[data[attribute] <= info[2]].sum()
        w = count / known_length
        adjusted_count = count + (weights[missing] * w).sum()

        distribution.append((count, adjusted_count))

        # upper partition
        count = weights[data[attribute] > info[2]].sum()
        w = count / known_length
        adjusted_count = count + (weights[missing] * w).sum()

        distribution.append((count, adjusted_count))
    else:
        attribute_values = list(
            filter(lambda v: v == v, data[attribute].unique()))

        for value in attribute_values:
            count = weights[data[attribute] == value].sum()
            w = count / known_length
            adjusted_count = count + (weights[missing] * w).sum()

            distribution.append((count, adjusted_count))

    # only adds an internal node if there are enough instances for at least two branches
    valid = 0

    for _, adjusted_count in distribution:
        if adjusted_count >= MINIMUM_INSTANCES:
            valid += 1

    node = None

    if valid < 2:
        # not enough instances on branches, need to select another attribute
        return build_decision_nodes(data.drop(columns=attribute),
                                    tree,
                                    parent_majority,
                                    weights)
    else:
        node = Node(attribute, parent=tree)

        if is_numeric_dtype(data[attribute]):
            # continuous threshold value
            threshold = info[2]

            # slice if the data where the value <= threshold (lower)
            partition = data[attribute] <= threshold

            updated_weights = weights.copy()
            updated_weights[missing] = updated_weights[missing] * \
                (distribution[0][0] / known_length)
            updated_weights = updated_weights[partition | missing]

            branch_data = data[partition | missing].reset_index(drop=True)
            child = build_decision_nodes(
                branch_data, node, majority, updated_weights)
            node.add(pre_prune(branch_data, child, updated_weights),
                     threshold,
                     Operator.LESS_OR_EQUAL)

            # slice if the data where the value > threshold (upper)
            partition = data[attribute] > threshold

            updated_weights = weights.copy()
            updated_weights[missing] = updated_weights[missing] * \
                (distribution[1][0] / known_length)
            updated_weights = updated_weights[partition | missing]

            branch_data = data[partition | missing].reset_index(drop=True)
            child = build_decision_nodes(
                branch_data, node, majority, updated_weights)
            node.add(pre_prune(branch_data, child, updated_weights),
                     threshold,
                     Operator.GREATER)
        else:
            # categorical split
            for index, value in enumerate(attribute_values):
                partition = data[attribute] == value

                updated_weights = weights.copy()
                updated_weights[missing] = updated_weights[missing] * \
                    (distribution[index][0] / known_length)
                updated_weights = updated_weights[partition | missing]

                branch_data = data[partition | missing].drop(
                    columns=attribute).reset_index(drop=True)
                child = build_decision_nodes(
                    branch_data, node, majority, updated_weights)
                node.add(pre_prune(branch_data, child,
                         updated_weights), value, Operator.EQUAL)

        # checks whether to prune the node or not
        if PRUNE:
            node = post_prune(data, node, majority, weights)

    # if we are the root node of the tree
    if tree is None:
        node.adjust(data)
        node.sort()
        node._classes = class_attribute.unique()

    return node

#
# Loaders
# ---------


def load_arff(path, class_attribute=None):
    """Loads the specified ARFF file into a Pandas DataFrame

    Parameters
    ----------
    path : str
        Path to the ARFF file

    Returns
    -------
    DataFrame
        DataFrame representing the content of the ARFF file
    """

    # attribute identifier
    ATTRIBUTE = "@attribute"
    # data identifier
    DATA = "@data"
    # separator identifier
    SEPARATOR = ","

    columns = []    # list of attributes
    data_types = []  # data type of the attributes

    # opens the file for reading
    file = open(path, "r")
    content = False
    rows = []

    for line in file:
        if content:
            values = line.split(SEPARATOR)
            rows.append([v.strip() for v in values])

        # @attribute
        elif ATTRIBUTE in line:
            start = line.index(' ', line.index(ATTRIBUTE)) + 1
            end = line.index(' ', start + 1)
            columns.append(line[start:end].strip())

            if "{" in line and "}" in line:
                data_types.append(str)
            else:
                data_types.append(float)

        # @data
        elif DATA in line:
            content = True

    # creating the pandas representation
    data = {}

    for index, data_type in enumerate(data_types):
        if data_type == str:
            data[columns[index]] = [np.nan if v[index] == '?' else v[index]
                                    for v in rows]
        elif data_type == float:
            data[columns[index]] = [np.nan if v[index] ==
                                    '?' else float(v[index]) for v in rows]

    class_attribute = columns[-1] if class_attribute is None else class_attribute
    class_column = data.pop(class_attribute)
    data = pd.DataFrame(data)
    data[class_attribute] = class_column

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='P45: A Python C4.5 implementation.')

    parser.add_argument('training',
                        type=str,
                        metavar='<training file>',
                        help='training file')

    parser.add_argument('-m',
                        type=int,
                        metavar='cases',
                        default=2,
                        help='minimum number of cases per leaf node')

    parser.add_argument('--prune',
                        type=bool,
                        default=True,
                        help='enables/disables pruning')

    parser.add_argument('-t',
                        type=str,
                        metavar='<test file>',
                        dest='test',
                        help='test file')

    args = parser.parse_args()
    # sets the algorithm parameters
    PRUNE = args.prune
    MINIMUM_INSTANCES = args.m

    print("P45 [Release 1.0]")
    print(datetime.today().strftime("%a %b %d %H:%M:%S %Y"))
    print("-----------------")

    data = load_arff(args.training)
    tree = build_decision_nodes(data)

    print("Decision tree:\n")
    tree.to_text()
    print("\n\nTime: 0.0 secs")
