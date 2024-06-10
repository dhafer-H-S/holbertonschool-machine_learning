#!/usr/bin/env python3

"""import numpy module"""
import numpy as np


class Node:
    """
    Represents a node in a decision tree.

    Attributes:
        feature (str): The feature used for splitting at this node.
        threshold (float): The threshold value for the feature used for
        splitting at this node.
        left_child (Node): The left child node.
        right_child (Node): The right child node.
        is_leaf (bool): Indicates whether this node is a leaf node.
        is_root (bool): Indicates whether this node is the root node.
        sub_population (None): Placeholder for storing the sub-population
        associated with this node.
        depth (int): The depth of this node in the decision tree.

    Methods:
        max_depth_below(): Returns the maximum depth of any leaf node below
        this node.
        count_nodes_below(only_leaves=False): Returns the number of nodes
        below this node.
        left_child_add_prefix(text): Adds a prefix to the text representation
        of the left child node.
        right_child_add_prefix(text): Adds a prefix to the text representation
        of the right child node.
        __str__(): Returns a string representation of the node.
    """

    def __init__(
            self,
            feature=None,
            threshold=None,
            left_child=None,
            right_child=None,
            is_root=False,
            depth=0):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        if self.is_leaf:
            return self.depth
        else:
            return max(
                self.left_child.max_depth_below(),
                self.right_child.max_depth_below())

    def count_nodes_below(self, only_leaves=False):
        if self.is_leaf:
            return 1
        else:
            left_count = self.left_child.count_nodes_below(
                only_leaves=only_leaves) if self.left_child else 0
            right_count = self.right_child.count_nodes_below(
                only_leaves=only_leaves) if self.right_child else 0
            if only_leaves:
                return right_count + left_count
            else:
                return 1 + right_count + left_count

    def left_child_add_prefix(self, text):
        lines = text.split("\n")
        new_text = "    +---> " + lines[0] + "\n"
        for x in lines[1:]:
            new_text += "    |  " + x + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        lines = text.split("\n")
        new_text = "    +---> " + lines[0] + "\n"
        for x in lines[1:]:
            new_text += "       " + x + "\n"
        return new_text

    def __str__(self):
        if self.is_root:
            node_str = "root [feature={}, threshold={}]".format(
                self.feature, self.threshold)
        else:
            node_str = "node [feature={}, threshold={}]".format(
                self.feature, self.threshold)

        if self.left_child:
            left_str = self.left_child_add_prefix(str(self.left_child)\
                                                  .strip())
        else:
            left_str = ""

        if self.right_child:
            right_str = self.right_child_add_prefix(
                str(self.right_child).strip())
        else:
            right_str = ""

        return "{}\n{}{}".format(node_str, left_str, right_str)


class Leaf(Node):
    """class leaf"""
    def __init__(self, value, depth=None):
        """
        Initializes a DecisionTreeNode object.

        Args:
            value (any): The value associated with the node.
            depth (int, optional): The depth of the node in the decision tree. Defaults to None.
        """
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """
        Returns the maximum depth below the current leaf node.

        Returns:
            int: The maximum depth below the current leaf node.
        """
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """
        Counts the number of nodes below the current leaf node.

        Args:
            only_leaves (bool, optional): If True,
            counts only the leaf nodes
            Defaults to False.

        Returns:
            int: The number of nodes below the current leaf node.
        """
        return 1

    def __str__(self):
        """
        Returns a string representation of the leaf node.

        Returns:
            str: A string representation of the leaf node.
        """
        return (f"leaf [value={self.value}] ")

    def get_leaves_below(self):
        """
        Returns a list of leaf nodes below the current leaf node.

        Returns:
            list: A list of leaf nodes below the current leaf node.
        """
        return [self]

    def __doc__(self):
        """
        Represents a leaf node in a decision tree.

        Attributes:
            value (any): The value associated with the leaf node.
            depth (int, optional): The depth of the leaf node in the tree.
        """
        pass


class Decision_Tree():
    """
    A class representing a decision tree.

    Attributes:
        max_depth (int): The maximum depth of the decision tree.
        Default is 10.
        min_pop (int): The minimum population required to create a split.
        Default is 1.
        seed (int): The seed value for random number generation.
        Default is 0.
        split_criterion (str): The criterion used for splitting the tree.
        Default is "random".
        root (Node): The root node of the decision tree. If not provided,
        a new root node will be created.
        explanatory: The explanatory variables used for training
        the decision tree.
        target: The target variable used for training the decision tree.
        predict: The function used for making predictions with
        the decision tree.
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

    def count_nodes(self, only_leaves=False):
        """
        Returns the number of nodes in the decision tree.

        Args:
            only_leaves (bool): If True, only counts the leaf nodes.
            Default is False.

        Returns:
            int: The number of nodes in the decision tree.
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """
        Returns a string representation of the decision tree.

        Returns:
            str: A string representation of the decision tree.
        """
        return self.root.__str__()

    def get_leaves(self):
        """
        Returns a list of leaf nodes in the decision tree.

        Returns:
            list: A list of leaf nodes in the decision tree.
        """
        return self.root.get_leaves_below()
