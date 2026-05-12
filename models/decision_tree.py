import copy
import math
import random
import numpy as np
from typing import Callable, Optional
from metrics import *

class Node:
    """
    Class to define a node in the decision tree.
    """
    def __init__(
            self,
            left: Optional["Node"] = None,
            right: Optional["Node"] = None,
            index_split_on: Optional[int] = None,
            threshold: Optional[float] = None,
            depth: int = 0,
            isleaf: bool = False,
            value: float = 1.0
            ) -> None:
        """Constructor for a node.

        Attributes
        ----------
        left : Optional["Node"]
            Optional left child node, by default None
        right : Optional["Node"]
            Optional right child node, by default None
        depth : int, optional
            Indication of current node depth in tree, by default 0
        index_split_on : int, optional
            Indication of index to split on, by default 0
        isleaf : bool, optional
            Should be true if the current node is a leaf, by default False
        label : int, optional
            Label value for current node, by default 1
        info : dict
            Dictionary to track gain values for visualization
        """
        self.left = left
        self.right = right
        self.index_split_on = index_split_on
        self.threshold = threshold
        self.depth = depth
        self.isleaf = isleaf
        self.value = value
        self.info = {}
    
    def _set_info(self, gain: float, num_samples: int, threshold: Optional[float] = None) -> None:
        """
        Store node states and metrics.
        """
        self.info["gain"] = gain
        self.info["num_samples"] = num_samples
        self.info["threshold"] = threshold

class DecisionTree:
    """Class defining a complete decision tree model"""
    def __init__(
            self,
            data: np.array,
            validation_data: np.ndarray = None,
            gain_function: Callable = entropy,
            max_depth: int = 40,
            min_samples: int = 2,
            max_features: Optional[int] = None           
    ) -> None:
        """Constructor

        Parameters
        ----------
        data : np.ndarray
            Training dataset
        validation_data : np.ndarray, optional
            Validation dataset, by default None
        gain_function : Callable, optional
            Function pointer to use as gain function, by default entropy
        max_depth : int, optional
            The maximum depth the tree is allowed to grow to prevent overfitting, by default 40.
        min_samples : int, optional
            The minimum number of samples required in a node to attempt a split, by default 2.
        max_features : int, optional
            The number of features to consider when looking for the best split, by default None.
        """

        self.gain_function = gain_function
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.max_features = max_features
        self.root = Node()

        indices = list(range(1, len(data[0])))
        self._split_recurs(self.root, data, indices)

        # Pruning
        if validation_data is not None:
            self._prune_recurs(self.root, validation_data)

    # ---------------------------------------------------------
    # METRICS
    # ---------------------------------------------------------

    def loss(self, data: np.ndarray):
        y_pred = self.predict(data[:, 1:])
        return np.mean(y_pred != data[:, 0])

    def accuracy(self, data: np.ndarray):
        y_pred = self.predict(data[:, 1:])
        return np.mean(y_pred == data[:, 0])
    
    # ---------------------------------------------------------
    # TRAINING PHASE
    # ---------------------------------------------------------

    def _calc_gain(
        self,
        data: np.ndarray,
        split_index: int,
        threshold: Optional[float],
        gain_function: Callable[[float], float],
    ) -> float:
        """
        Calculate the gain of the proposed splitting.

        Gain = C(P[y=1]) - P[x_i < threshold] * C(P[y=1|x_i < threshold]) - P[x_i >= threshold] * C(P[y=1|x_i >= threshold])

        C(p) is the gain_function defined in the metrics.py

        Parameters
        ----------
        data : np.ndarray
           Numpy array with size of n*(m_1 + m_2 + ... + m_m + 1), the first column is 1 or 0 corresponding to label
        split_index : int
            Int representing the index to split on.
        threshold: float
            Float representing the threshold to split
        gain_function: function
            Function pointer to gain function (float to float) which calculates the node score. One of
            error, entropy, or gini in metrics.py

        Returns
        -------
        float
            The gain value of the proposed splitting.
        """

        # P[x_i < threshold]
        left_data = data[data[:, split_index] < threshold]
        w_left = len(left_data) / len(data)
        # P[x_i >= threshold]
        right_data = data[data[:, split_index] >= threshold]
        w_right = len(right_data) / len(data)

        if len(left_data) == 0 or len(right_data) == 0:
            return 0.0

        # calculate gain
        gain = gain_function(np.mean(data[:, 0])) \
                - w_left * gain_function(np.mean(left_data[:, 0])) \
                - w_right * gain_function(np.mean(right_data[:, 0]))
        return gain
    
    def _is_terminal(
        self, node: Node, data: np.ndarray, indices: list[int]
        ) -> tuple[bool, int]:
        """
        Determines whether or not the node should stop splitting. Stop the recursion
        in the following cases -
            1. The dataset is empty.
            2. There are no more indices to split on.
            3. All the instances in this dataset belong to the same class
            4. The depth of the node exceeds the maximum depth.

        Parameters
        ----------
        node : Node
           Current node
        data : np.ndarray
           Numpy array with size of n*(m_1 + m_2 + ... + m_m+1), the first column is 1 or 0 corresponding to label
        indices : list
            List of indices to split on

        Returns
        -------
        tuple[bool, int]
            A tuple consisting of a boolean in the first index and an integer label in the second.
            - For the boolean, True indicates that the passed Node should be a leaf; return False otherwise.
            - For the integer label, indicate the leaf label by majority or the label the passed Node would
              have if we were to terminate at it instead. If there is no data left (i.e., len(data) == 0),
              return a label randomly (see eg. numpy.random.choice())
        """
        labels = data[:, 0]
        # Empty data -> return random label
        if len(data) == 0:
            return True, int(np.random.choice([0, 1]))
        
        # Reach purity -> return the label
        unique_labels = np.unique(labels)
        if len(unique_labels) == 1:
            return True, int(unique_labels[0])
        
        majority_label = int(np.round(np.mean(labels)))
        # No more indices
        if len(indices) == 0:
            return True, majority_label
        
        # Reach max depth
        if node.depth >= self.max_depth:
            return True, majority_label
        
        # Reach min sample
        if len(data) < self.min_samples:
            return True, majority_label
        
        return False, majority_label

    def _split_recurs(self, node: Node, data: np.ndarray, indices: list[int]) -> None:
        """
        Recursively split the node based on the rows and indices given.
        Nothing needs to be returned.

        First, check if the node needs to be split (see _is_terminal()).  If so, find the
        optimal column to split on by maximizing information gain, ie, with _calc_gain().
        Store the label predicted for this node, the split column, and use _set_info().
        Then, split the data based on its value in the selected column.
        The data should be recursively passed to the children.

        Parameters
        ----------
        node : Node
           Current node
        data : np.ndarray
           Numpy array with size of n*(m_1 + m_2 + ... + m_m + 1), the first column is 1 or 0 corresponding to label
        indices : list
            List of indices to split on

        Returns
        -------
        None
        """
        is_term, node_value = self._is_terminal(node, data, indices)
        node.value = np.round(node_value)

        if is_term:
            node.isleaf = True
            return
        
        best_gain = -1
        best_idx = None
        best_thr = None

        for i in indices:
            unique_values = np.unique(data[:, i])
            thresholds = (unique_values[:-1] + unique_values[1:]) /2
            for thr in thresholds:
                current_gain = self._calc_gain(data, i, thr, self.gain_function)
                if current_gain > best_gain:
                    best_gain = current_gain
                    best_idx = i
                    best_thr = thr
        
        if best_idx is None:
            node.isleaf = True
            return
        
        node.index_split_on = best_idx
        node.threshold = best_thr
        node._set_info(gain=best_gain,
                       num_samples=len(data))
        
        left_subset = data[data[:, best_idx] < best_thr]
        right_subset = data[data[:, best_idx] >= best_thr]

        node.left = Node(depth=node.depth + 1)
        node.right = Node(depth=node.depth + 1)

        new_indices = [i for i in indices if i != best_idx]

        self._split_recurs(node.left, left_subset, new_indices)
        self._split_recurs(node.right, right_subset, new_indices)

    # ---------------------------------------------------------
    # PRUNING
    # ---------------------------------------------------------

    def _prune_recurs(self, node: Node, validation_data: np.ndarray) -> None:
        """
        Prunes the tree bottom up recursively. DO NOT prune if the node is a leaf
        or if the node is non-leaf and has at least one non-leaf child. On the other hand,
        DO prune if doing so could reduce loss on the validation data.
        
        Parameters
        ----------
        node : Node
           Current node
        validation_data : np.ndarray
           Numpy array with size of n*(m_1 + m_2 + ... + m_m+1), the first column
           is 1 or 0 corresponding to label

        Returns
        -------
        None
        """
        pruned_error = np.sum(validation_data[:, 0] != node.value)

        if node.isleaf:
            return pruned_error
        
        left_val_subset = validation_data[validation_data[:, node.index_split_on] < node.threshold]
        right_val_subset = validation_data[validation_data[:, node.index_split_on] >= node.threshold]

        current_error =  self._prune_recurs(node.left, left_val_subset) + self._prune_recurs(node.right, right_val_subset)

        if pruned_error < current_error:
            node.isleaf = True
            node.left = None
            node.right = None
            return pruned_error
        else:
            return current_error
        
    # ---------------------------------------------------------
    # FIT
    # ---------------------------------------------------------

    def fit(self, train_data: np.ndarray):
        """
        The public entry point to train the decision tree.
        """
        # 1. Initialize the root node at depth 0
        self.root = Node(depth=0)
        
        # 2. Identify the feature indices (all columns except the label at index 0)
        # data.shape[1] gives us the number of columns
        all_indices = list(range(1, train_data.shape[1]))
        
        # 3. Kick off the recursion!
        self._split_recurs(self.root, train_data, all_indices)
        
    # ---------------------------------------------------------
    # INFERENCE
    # ---------------------------------------------------------

    def predict(self, data: np.ndarray):
        """
        Predict labels for a dataset.
        data: n x m matrix (no labels, just features)
        """
        return np.array([self._predict_recurs(self.root, row) for row in data])
    
    def _predict_recurs(self, node: Node, row: np.ndarray, threshold: float):
        """
        Helper function to predict a single row by traversing the tree.
        """
        if node.isleaf:
            return node.value
        if row[node.index_split_on] < node.threshold:
            return self._predict_recurs(node.left, row)
        else:
            return self._predict_recurs(node.right, row)
    
    # ---------------------------------------------------------
    # TREE & LOSS VISUALIZATION
    # ---------------------------------------------------------

    def print_tree(self, node: Node = None, indent: str = ""):
        """
        Visualizes the tree structure in the console.
        """
        # Start at the root if no node is provided
        if node is None:
            node = self.root

        # Base Case: It's a leaf
        if node.isleaf:
            print(f"{indent}Leaf: Prediction = {node.value}")
            return

        # Recursive Step: Print current split and move to children
        print(f"{indent}[Feature {node.index_split_on} < {node.threshold:.4f}]")
        
        print(f"{indent}  L:", end="")
        self.print_tree(node.left, indent + "    ")
        
        print(f"{indent}  R:", end="")
        self.print_tree(node.right, indent + "    ")

    def _loss_plot_recurs(self, node: Node, data: np.ndarray, losses: list):
        """
        Recursively calculates loss at each node and stores it for plotting.
        """
        # Calculate the loss for the current node's prediction
        # This shows what the error would be if we stopped here
        current_loss = np.mean(data[:, 0] != node.value)
        losses.append(current_loss)

        # If it's a leaf, we can't go any deeper
        if node.isleaf:
            return

        # Split data to continue the journey    
        left_mask = data[:, node.index_split_on] < node.threshold
        
        # Only recurse if there is data following that path
        if np.any(left_mask):
            self._loss_plot_recurs(node.left, data[left_mask], losses)
        if np.any(~left_mask):
            self._loss_plot_recurs(node.right, data[~left_mask], losses)

    def _loss_plot_vec(self, data: np.ndarray):
        """
        Generates a vector of losses across the tree structure.
        """
        losses = []
        # We pass the full data (including labels) to calculate error
        self._loss_plot_recurs(self.root, data, losses)
        
        # Returning as a numpy array makes it easy to plot or average
        return np.array(losses)