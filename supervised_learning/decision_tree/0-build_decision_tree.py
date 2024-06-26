#!/usr/bin/env python3

"""import numpy module"""
import numpy as np


class Node:
    """
    This class represents a node in a decision tree.

    A node holds the feature and threshold used for splitting the data,
    references to the left and right child nodes, and other properties
    related to the node's role in the tree.

    Attributes:
        feature (int): The index of the feature used for
        splitting the data at this node.
        threshold (float): The threshold value used for splitting the
        data at this node.
        left_child (Node): The left child node.
        right_child (Node): The right child node.
        is_root (bool): Indicates whether this node is the root of
        the decision tree.
        depth (int): The depth of the node in the decision tree.
    """

    def __init__(
            self,
            feature=None,
            threshold=None,
            left_child=None,
            right_child=None,
            is_root=False,
            depth=0):
        """
        Initialize a DecisionTreeNode object.

        Args:
            feature (int): The index of the feature used for splitting
            the data at this node.
            threshold (float): The threshold value used for splitting
            the data at this node.
            left_child (DecisionTreeNode): The left child node.
            right_child (DecisionTreeNode): The right child node.
            is_root (bool): Indicates whether this node is the root of
            the decision tree.
            depth (int): The depth of the node in the decision tree.

        Attributes:
            feature (int): The index of the feature used for splitting
            the data at this node.
            threshold (float): The threshold value used for splitting
            the data at this node.
            left_child (DecisionTreeNode): The left child node.
            right_child (DecisionTreeNode): The right child node.
            is_leaf (bool): Indicates whether this node is a leaf node.
            is_root (bool): Indicates whether this node is the root of
            the decision tree.
            sub_population (None): Placeholder for storing the subset
            of data at this node.
            depth (int): The depth of the node in the decision tree.
        """
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """
        Calculate the maximum depth below this node.

        Returns:
            int: The maximum depth below this node.
        """
        if self.is_leaf:
            return self.depth
        else:
            return max(
                self.left_child.max_depth_below(),
                self.right_child.max_depth_below())


class Leaf(Node):
    """
    Represents a leaf node in a decision tree.

    Attributes:
        value (any): The value associated with the leaf node.
        is_leaf (bool): Indicates whether the node is a leaf node.
        depth (int): The depth of the leaf node in the decision tree.
    """

    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """
        Returns the maximum depth below the leaf node.

        Returns:
            int: The maximum depth below the leaf node.
        """
        return self.depth


class Decision_Tree():
    """
    Decision_Tree class represents a decision tree model.

    Attributes:
        max_depth (int): The maximum depth of the decision tree. Default is 10.
        min_pop (int): The minimum number of samples required to split a node.
        Default is 1.
        seed (int): The seed value for random number generation. Default is 0.
        split_criterion (str): The criterion used to split the nodes.
        Default is "random".
        root (Node): The root node of the decision tree. Default is None.

    Methods:
        depth(): Returns the maximum depth of the decision tree.
    """

    def __init__(
            self,
            max_depth=10,
            min_pop=1,
            seed=0,
            split_criterion="random",
            root=None):
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """
        Returns the maximum depth of the decision tree.

        Returns:
            int: The maximum depth of the decision tree.
        """
        return self.root.max_depth_below()
