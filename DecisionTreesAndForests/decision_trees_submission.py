from __future__ import division

import numpy as np
from collections import Counter
import time


class DecisionNode:
    """Class to represent a single node in a decision tree."""

    def __init__(self, left, right, decision_function, class_label=None):
        """Create a decision function to select between left and right nodes.

        Note: In this representation 'True' values for a decision take us to
        the left. This is arbitrary but is important for this assignment.

        Args:
            left (DecisionNode): left child node.
            right (DecisionNode): right child node.
            decision_function (func): function to decide left or right node.
            class_label (int): label for leaf node. Default is None.
        """

        self.left = left
        self.right = right
        self.decision_function = decision_function
        self.class_label = class_label

    def decide(self, feature):
        """Get a child node based on the decision function.

        Args:
            feature (list(int)): vector for feature.

        Return:
            Class label if a leaf node, otherwise a child node.
        """
        feature = list(feature)
        if self.class_label is not None:
            return self.class_label

        elif self.decision_function(feature):
            return self.left.decide(feature)

        else:
            return self.right.decide(feature)


def load_csv(data_file_path, class_index=-1):
    """Load csv data in a numpy array.

    Args:
        data_file_path (str): path to data file.
        class_index (int): slice output by index.

    Returns:
        features, classes as numpy arrays if class_index is specified,
            otherwise all as nump array.
    """

    handle = open(data_file_path, 'r')
    contents = handle.read()
    handle.close()
    rows = contents.split('\n')
    out = np.array([[float(i) for i in r.split(',')] for r in rows if r])

    if class_index == -1:
        classes = map(int, out[:, class_index])
        features = out[:, :class_index]
        return features, classes

    elif class_index == 0:
        classes = map(int, out[:, class_index])
        features = out[:, 1:]
        return features, classes

    else:
        return out


def build_decision_tree():
    """Create a decision tree capable of handling the provided data.

    Tree is built fully starting from the root.

    Returns:
        The root node of the decision tree.
    """

    decision_tree_root = None

    # TODO: finish this.
    #raise NotImplemented()

    decision_tree_root = DecisionNode(None, None, lambda feature: feature[0] == 0)
    decision_tree_root.left = DecisionNode(None, None, lambda feature: feature[2] == 0)
    decision_tree_root.right = DecisionNode(None, None, None, 1)

    a3 = decision_tree_root.left
    a3.left = DecisionNode(None, None, lambda feature: feature[3] == 0)
    a3.right = DecisionNode(None, None, lambda feature: feature[3] == 0)

    a4left = a3.left
    a4left.left = DecisionNode(None, None, None, 1)
    a4left.right = DecisionNode(None, None, None, 0)

    a4right = a3.right
    a4right.left = DecisionNode(None, None, None, 0)
    a4right.right = DecisionNode(None, None, None, 1)

    return decision_tree_root

def confusion_matrix(classifier_output, true_labels):
    """Create a confusion matrix to measure classifier performance.

    Output will in the format:
        [[true_positive, false_negative],
         [false_positive, true_negative]]

    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.

    Returns:
        A two dimensional array representing the confusion matrix.
    """

    # TODO: finish this.
    #raise NotImplemented()
    classifier_output = np.array(classifier_output)
    true_labels = np.array(true_labels)
    match_list = true_labels[classifier_output == true_labels]
    non_match_list = true_labels[classifier_output != true_labels]
    TP = np.sum(np.array(match_list) == 1).astype(float)
    TN = np.sum(np.array(match_list) == 0).astype(float)
    FN = np.sum(np.array(non_match_list) == 1).astype(float)
    FP = np.sum(np.array(non_match_list) == 0).astype(float)
    conf_matrix = np.array([[TP, FN], [FP, TN]])
    return conf_matrix


def precision(classifier_output, true_labels):
    """Get the precision of a classifier compared to the cclassesorrect values.

    Precision is measured as:
        true_positive/ (true_positive + false_positive)

    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.

    Returns:
        The precision of the classifier output.
    """

    # TODO: finish this.
    #raise NotImplemented()
    classifier_output = np.array(classifier_output)
    true_labels = np.array(true_labels)
    match_list = true_labels[classifier_output == true_labels]
    non_match_list = true_labels[classifier_output != true_labels]
    TP = np.sum(np.array(match_list) == 1).astype(float)
    TN = np.sum(np.array(match_list) == 0).astype(float)
    FN = np.sum(np.array(non_match_list) == 1).astype(float)
    FP = np.sum(np.array(non_match_list) == 0).astype(float)
    return TP/(TP+FP)


def recall(classifier_output, true_labels):
    """Get the recall of a classifier compared to the correct values.

    Recall is measured as:
        true_positive/ (true_positive + false_negative)

    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.

    Returns:
        The recall of the classifier output.
    """

    # TODO: finish this.
    #raise NotImplemented()
    classifier_output = np.array(classifier_output)
    true_labels = np.array(true_labels)
    match_list = true_labels[classifier_output == true_labels]
    non_match_list = true_labels[classifier_output != true_labels]
    TP = np.sum(np.array(match_list) == 1).astype(float)
    TN = np.sum(np.array(match_list) == 0).astype(float)
    FN = np.sum(np.array(non_match_list) == 1).astype(float)
    FP = np.sum(np.array(non_match_list) == 0).astype(float)
    #print(match_list, non_match_list, TP, FN)
    return TP/(TP+FN)


def accuracy(classifier_output, true_labels):
    """Get the accuracy of a classifier compared to the correct values.

    Accuracy is measured as:
        correct_classifications / total_number_examples

    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.

    Returns:
        The accuracy of the classifier output.
    """

    # TODO: finish this.
    #raise NotImplemented()
    num_matches = np.sum(np.array(classifier_output) == np.array(true_labels)).astype(float)
    _accuracy = num_matches/len(true_labels)
    return _accuracy

def gini_impurity(class_vector):
    """Compute the gini impurity for a list of classes.
    This is a measure of how often a randomly chosen element
    drawn from the class_vector would be incorrectly labeled
    if it was randomly labeled according to the distribution
    of the labels in the class_vector.
    It reaches its minimum at zero when all elements of class_vector
    belong to the same class.

    Args:
        class_vector (list(int)): Vector of classes given as 0 or 1.

    Returns:
        Floating point number representing the gini impurity.
    """
    #raise NotImplemented()

    classes = np.array(class_vector)
    total = len(class_vector)
    gini = 1.0 - (np.sum(classes == 0).astype(float)/total)**2 - (np.sum(classes == 1).astype(float)/total)**2
    return gini


def gini_gain(previous_classes, current_classes):
    """Compute the gini impurity gain between the previous and current classes.
    Args:
        previous_classes (list(int)): Vector of classes given as 0 or 1.
        current_classes (list(list(int): A list of lists where each list has
            0 and 1 values).
    Returns:
        Floating point number representing the information gain.
    """
    #raise NotImplemented()
    gini = gini_impurity(previous_classes)
    total = np.float(len(previous_classes))
    for i in range(len(current_classes)):
        curr_len = np.float(len(current_classes[i]))
        gini = gini - gini_impurity(current_classes[i]) * curr_len / total

    return gini



class DecisionTree:
    """Class for automatic tree-building and classification."""

    def __init__(self, depth_limit=float('inf')):
        """Create a decision tree with a set depth limit.

        Starts with an empty root.

        Args:
            depth_limit (float): The maximum depth to build the tree.
        """

        self.root = None
        self.depth_limit = depth_limit

    def fit(self, features, classes):
        """Build the tree from root using __build_tree__().

        Args:
            features (list(list(int)): List of features.
            classes (list(int)): Available classes.
        """

        self.root = self.__build_tree__(features, classes)

    def __build_tree__(self, features, classes, depth=0):
        """Build tree that automatically finds the decision functions.

        Args:
            features (list(list(int)): List of features.
            classes (list(int)): Available classes.
            depth (int): max depth of tree.  Default is 0.

        Returns:
            Root node of decision tree.
        """

        # TODO: finish this.
        #raise NotImplemented()
        #features, classes = load_csv('part23_data.csv', -1)
        classes = np.array(classes)
        features = np.array(features)
        #print(classes.shape, features.shape)
        #if (len(classes) == 0):
        #    return None
        #print(classes.shape, len(set(classes.flatten())))
        if(len(set(classes)) == 1):
            #print("haha")
            return DecisionNode(None, None, None, classes[0])

        if depth == self.depth_limit:
            count_class_0 = np.sum(classes == 0)
            count_class_1 = np.sum(classes == 1)
            if count_class_1 > count_class_0:
                return DecisionNode(None, None, None, 1)
            else:
                return DecisionNode(None, None, None, 0)
        else:
            best_feature_index = 0
            best_gini = float('-inf')
            final_threshold = float('-inf')
            final_class_distribution = []
            #print(features.shape)
            for i in range(len(features[0])):

                feature_max = max(features[:, i])
                feature_min = min(features[:, i])
                #print(feature_min, feature_max)
                # 200 steps for finding the right threshold
                step = (feature_max - feature_min)/200.0
                best_class_distribution = []
                best_threshold = float('-inf')
                best_ig = float('-inf')
                if(set(classes) == 1):
                    return DecisionNode(None, None, None, classes[0])

                for j in np.arange(feature_min + step, feature_max, step):
                    curr_threshold = j
                    # class distribution of left and right child, 0 left, 1 right
                    class_distribution = np.zeros(len(classes))
                    #print(class_distribution.shape)
                    for k in range(len(classes)):
                        if features[k, i] > curr_threshold:
                            class_distribution[k] = 1

                    gini = gini_gain(classes, [classes[np.where(class_distribution == 0)],
                                                       classes[np.where(class_distribution == 1)]])
                    #print(gini)
                    if gini > best_ig:
                        best_ig = gini
                        best_threshold = curr_threshold
                        best_class_distribution = class_distribution
                if best_gini < best_ig:
                    best_gini = best_ig
                    best_feature_index = i
                    final_threshold = best_threshold
                    final_class_distribution = best_class_distribution

            #print(final_threshold, best_feature_index, best_gini)
            if best_gini == 0.0:
                return DecisionNode(None, None, None, classes[0])
            #print(len(final_class_distribution))
            class_vector_left = classes[np.where(features[:, best_feature_index] <= final_threshold)]
            features_left = features[np.where(features[:, best_feature_index] <= final_threshold)]
            class_vector_right = classes[np.where(features[:, best_feature_index] > final_threshold)]
            features_right = features[np.where(features[:, best_feature_index] > final_threshold)]
            #print(class_vector_left.shape, features_left.shape, class_vector_right.shape, features_right.shape)

            curr_node = DecisionNode(None, None, lambda features: features[best_feature_index] <= final_threshold)
            curr_node.left = self.__build_tree__(features_left, class_vector_left, depth = depth + 1)
            curr_node.right = self.__build_tree__(features_right, class_vector_right, depth = depth + 1)

            return curr_node

    def classify(self, features):
        """Use the fitted tree to classify a list of example features.

        Args:
            features (list(list(int)): List of features.

        Return:
            A list of class labels.
        """

        class_labels = []

        # TODO: finish this.
        # raise NotImplemented()
        class_labels = []
        for i in range(len(features)):
            class_labels.append(self.root.decide(features[i]))


        return class_labels


def generate_k_folds(dataset, k):
    """Split dataset into folds.

    Randomly split data into k equal subsets.

    Fold is a tuple (training_set, test_set).
    Set is a tuple (examples, classes).

    Args:
        dataset: dataset to be split.
        k (int): number of subsections to create.

    Returns:
        List of folds.
    """

    # TODO: finish this.
    #raise NotImplemented()
    examples = np.array(dataset[0])
    classes = np.array(dataset[1]).reshape(-1,1)
    num_examples = len(examples)
    examples_classes = np.hstack((examples, classes))
    np.random.shuffle(examples_classes)
    num_examples_per_step = num_examples//k
    k_folds = []
    for i in range(k):
        begin_idx = i*num_examples_per_step
        end_idx = begin_idx + num_examples_per_step

        test_fold_idx = range(begin_idx, end_idx)
        #print(test_fold_idx)
        test_fold = examples_classes[test_fold_idx]
        test_fold_examples = test_fold[:, :-1]
        test_fold_classes = test_fold[:, -1]

        train_fold_idx = np.concatenate((range(0, begin_idx),range(end_idx, num_examples))).astype(int)

        train_fold = examples_classes[train_fold_idx]
        train_fold_examples = train_fold[:, :-1]
        train_fold_classes = train_fold[:, -1]
        #print(len(train_fold_idx))
        k_folds.append(((train_fold_examples, train_fold_classes), (test_fold_examples, test_fold_classes)))

    return k_folds

class RandomForest:
    """Random forest classification."""

    def __init__(self, num_trees, depth_limit, example_subsample_rate,
                 attr_subsample_rate):
        """Create a random forest.

         Args:
             num_trees (int): fixed number of trees.
             depth_limit (int): max depth limit of tree.
             example_subsample_rate (float): percentage of example samples.
             attr_subsample_rate (float): percentage of attribute samples.
        """

        self.trees = []
        self.num_trees = num_trees
        self.depth_limit = depth_limit
        self.example_subsample_rate = example_subsample_rate
        self.attr_subsample_rate = attr_subsample_rate
        self.sub_feat_idx = []

    def fit(self, features, classes):
        """Build a random forest of decision trees using Bootstrap Aggregation.

            features (list(list(int)): List of features.
            classes (list(int)): Available classes.
        """

        # TODO: finish this.
        #raise NotImplemented()
        classes = np.array(classes)
        features = np.array(features)
        #print(classes.shape)
        num_subsamples = int(self.example_subsample_rate * len(classes))
        num_features = int(self.attr_subsample_rate * len(features[0]))
        for i in range(self.num_trees):
            curr_subsample_features_idx= np.random.choice(len(classes), num_subsamples, replace = True).reshape(-1, 1)
            #print(curr_subsample_features_idx.shape, len(classes), len(features))
            curr_subsamp = features[curr_subsample_features_idx]
            curr_subsamp = np.squeeze(curr_subsamp)
            #print(curr_subsamp.shape)
            curr_subsample_subfeatures_idx = np.random.choice(len(features[0]), num_features, replace = False)
            self.sub_feat_idx.append(curr_subsample_subfeatures_idx)
            #classes = np.array(classes)
            curr_subsamp_subfeat = curr_subsamp[:, curr_subsample_subfeatures_idx]
            curr_subsamp_subfeat_classes = classes[curr_subsample_features_idx]

            curr_tree = DecisionTree(self.depth_limit)
            #print(curr_subsamp_subfeat.shape, curr_subsamp_subfeat_classes.shape)
            curr_tree.fit(curr_subsamp_subfeat, np.squeeze(curr_subsamp_subfeat_classes))


            self.trees.append(curr_tree)



    def classify(self, features):
        """Classify a list of features based on the trained random forest.

        Args:
            features (list(list(int)): List of features.
        """

        # TODO: finish this.
        #raise NotImplemented()
        results = []
        for i in range(self.num_trees):
            curr_tree = self.trees[i]
            class_labels = curr_tree.classify(features[:, self.sub_feat_idx[i]])
            class_labels = np.array(class_labels)
            #print(class_labels.shape)
            results.append( np.array(class_labels).reshape(-1, 1))

        results = np.column_stack(results)
        #print(results.shape)
        vote_label = np.sum(results, axis = 1).astype(np.float32)/self.num_trees
        #print(vote_label)
        final_label = [1 if i > 0.5 else 0 for i in vote_label]
        final_label = np.array(final_label).reshape(-1, 1)
        #print(final_label.shape)
        return final_label


class ChallengeClassifier:
    """Challenge Classifier used on Challenge Training Data."""

    def __init__(self, num_trees = 5, depth_limit = 5, example_subsample_rate = 0.7,
                 attr_subsample_rate = 0.7):
        """Create challenge classifier.

        Initialize whatever parameters you may need here.
        This method will be called without parameters, therefore provide
        defaults.
        """

        # TODO: finish this.
        #raise NotImplemented()

        self.trees = []
        self.num_trees = num_trees
        self.depth_limit = depth_limit
        self.example_subsample_rate = example_subsample_rate
        self.attr_subsample_rate = attr_subsample_rate
        self.sub_feat_idx = []

    def fit(self, features, classes):
        """Build the underlying tree(s).

            Fit your model to the provided features.

        Args:
            features (list(list(int)): List of features.
            classes (list(int)): Available classes.
        """

        # TODO: finish this.

        classes = np.array(classes)
        features = np.array(features)

        num_subsamples = int(self.example_subsample_rate * len(classes))
        num_features = int(self.attr_subsample_rate * len(features[0]))
        for i in range(self.num_trees):
            curr_subsample_features_idx= np.random.choice(len(classes), num_subsamples, replace = True).reshape(-1, 1)

            curr_subsamp = features[curr_subsample_features_idx]
            curr_subsamp = np.squeeze(curr_subsamp)

            curr_subsample_subfeatures_idx = np.random.choice(len(features[0]), num_features, replace = False)
            self.sub_feat_idx.append(curr_subsample_subfeatures_idx)

            curr_subsamp_subfeat = curr_subsamp[:, curr_subsample_subfeatures_idx]
            curr_subsamp_subfeat_classes = classes[curr_subsample_features_idx]

            curr_tree = DecisionTree(self.depth_limit)

            curr_tree.fit(curr_subsamp_subfeat, np.squeeze(curr_subsamp_subfeat_classes))


            self.trees.append(curr_tree)

    def classify(self, features):
        """Classify a list of features.

        Classify each feature in features as either 0 or 1.

        Args:
            features (list(list(int)): List of features.

        Returns:
            A list of class labels.
        """

        # TODO: finish this.

        results = []
        for i in range(self.num_trees):
            curr_tree = self.trees[i]
            class_labels = curr_tree.classify(features[:, self.sub_feat_idx[i]])
            class_labels = np.array(class_labels)

            results.append( np.array(class_labels).reshape(-1, 1))

        results = np.column_stack(results)

        vote_label = np.sum(results, axis = 1).astype(np.float32)/self.num_trees

        final_label = [1 if i > 0.5 else 0 for i in vote_label]
        final_label = np.array(final_label).reshape(-1, 1)

        return final_label


class Vectorization:
    """Vectorization preparation for Assignment 5."""

    def __init__(self):
        pass

    def non_vectorized_loops(self, data):
        """Element wise array arithmetic with loops.

        This function takes one matrix, multiplies by itself and then adds to
        itself.

        Args:
            data: data to be added to array.

        Returns:
            Numpy array of data.
        """

        non_vectorized = np.zeros(data.shape)
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                non_vectorized[row][col] = (data[row][col] * data[row][col] +
                                            data[row][col])
        return non_vectorized

    def vectorized_loops(self, data):
        """Element wise array arithmetic using vectorization.

        This function takes one matrix, multiplies by itself and then adds to
        itself.

        Bonnie time to beat: 0.09 seconds.

        Args:
            data: data to be sliced and summed.

        Returns:
            Numpy array of data.
        """

        # TODO: finish this.
        #raise NotImplemented()
        return np.multiply(data, data) + data


    def non_vectorized_slice(self, data):
        """Find row with max sum using loops.

        This function searches through the first 100 rows, looking for the row
        with the max sum. (ie, add all the values in that row together).

        Args:
            data: data to be added to array.

        Returns:
            Tuple (Max row sum, index of row with max sum)
        """

        max_sum = 0
        max_sum_index = 0
        for row in range(100):
            temp_sum = 0
            for col in range(data.shape[1]):
                temp_sum += data[row][col]

            if temp_sum > max_sum:
                max_sum = temp_sum
                max_sum_index = row

        return max_sum, max_sum_index

    def vectorized_slice(self, data):
        """Find row with max sum using vectorization.

        This function searches through the first 100 rows, looking for the row
        with the max sum. (ie, add all the values in that row together).

        Bonnie time to beat: 0.07 seconds

        Args:
            data: data to be sliced and summed.

        Returns:
            Tuple (Max row sum, index of row with max sum)
        """

        # TODO: finish this.
        #raise NotImplemented()
        sum_data = np.sum(data, axis = 1)[:100]
        max = np.max(sum_data)
        max_idx = np.argmax(sum_data)
        return max, max_idx


    def non_vectorized_flatten(self, data):
        """Display occurrences of positive numbers using loops.

         Flattens down data into a 1d array, then creates a dictionary of how
         often a positive number appears in the data and displays that value.

         ie, [(1203,3)] = integer 1203 appeared 3 times in data.

         Args:
            data: data to be added to array.

        Returns:
            List of occurrences [(integer, number of occurrences), ...]
        """

        unique_dict = {}
        flattened = np.hstack(data)
        for item in range(len(flattened)):
            if flattened[item] > 0:
                if flattened[item] in unique_dict:
                    unique_dict[flattened[item]] += 1
                else:
                    unique_dict[flattened[item]] = 1

        return unique_dict.items()

    def vectorized_flatten(self, data):
        """Display occurrences of positive numbers using vectorization.

         Flattens down data into a 1d array, then creates a dictionary of how
         often a positive number appears in the data and displays that value.

         ie, [(1203,3)] = integer 1203 appeared 3 times in data.

         Bonnie time to beat: 15 seconds

         Args:
            data: data to be added to array.

        Returns:
            List of occurrences [(integer, number of occurrences), ...]
        """

        # TODO: finish this.

        flattened = np.hstack(data)
        unique_num, count = np.unique(flattened, return_counts = True)
        unique_dict = [(unique_num[i], count[i]) for i in range(len(unique_num)) if unique_num[i] > 0 ]
        return unique_dict

        
def return_your_name():
    # return your name
    # TODO: finish this
    return "Xiangnan He"
