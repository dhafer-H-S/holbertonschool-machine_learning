#!/usr/bin/env python3

"""import numpy module"""
import numpy as np


class Node:
    """
    Represents a node in a decision tree.

    Attributes:
        feature (str): The feature used for splitting at this node.
        threshold (float): The threshold value for the feature.
        left_child (Node): The left child node.
        right_child (Node): The right child node.
        is_root (bool): Indicates if this node is the root of the tree.
        depth (int): The depth of the node in the tree.
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
        Initializes a DecisionTreeNode object.

        Args:
            feature (int): The index of the feature used for
            splitting at this node.
            threshold (float): The threshold value used for
            splitting at this node.
            left_child (DecisionTreeNode): The left child node.
            right_child (DecisionTreeNode): The right child node.
            is_root (bool): Indicates whether this node is the
            root of the decision tree.
            depth (int): The depth of the node in the decision tree.

        Returns:
            None
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
        Returns the maximum depth of the tree below this node.

        Returns:
            int: The maximum depth below this node.
        """
        if self.is_leaf:
            return self.depth
        else:
            return max(
                self.left_child.max_depth_below(),
                self.right_child.max_depth_below())

    def count_nodes_below(self, only_leaves=False):
        """
        Counts the number of nodes below this node.

        Args:
            only_leaves (bool): If True, only counts the leaf nodes.

        Returns:
            int: The number of nodes below this node.
        """
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
        """
        Adds a prefix to the text representation of the left child node.

        Args:
            text (str): The text representation of the left child node.

        Returns:
            str: The updated text representation with a prefix added.
        """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += "    |  " + x + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        """
        Adds a prefix to the text representation of the right child node.

        Args:
            text (str): The text representation of the right child node.

        Returns:
            str: The updated text representation with a prefix added.
        """
        lines = text.split("\n")
        new_text = "    `--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += "       " + x + "\n"
        return new_text

    def __str__(self):
        """
        Returns a string representation of the decision tree node.

        If the node is the root, it is represented as
        "root [feature=X, threshold=Y]".
        If the node is not the root, it is represented
        as "node [feature=X, threshold=Y]".

        The left child and right child nodes are recursively
        added to the string representation.

        Returns:
            str: A string representation of the decision tree node.
        """
        if self.is_root:
            node_str = "root [feature={}, threshold={}]".format(
                self.feature, self.threshold)
        else:
            node_str = "node [feature={}, threshold={}]".format(
                self.feature, self.threshold)

        if self.left_child:
            left_str = self.left_child_add_prefix(str(self.left_child))
        else:
            left_str = ""

        if self.right_child:
            right_str = self.right_child_add_prefix(str(self.right_child))
        else:
            right_str = ""

        return "{}\n{}{}".format(node_str, left_str, right_str)

    def get_leaves_below(self):
        """
        Returns a list of leaf nodes below this node.

        Returns:
            list: A list of leaf nodes below this node.
        """
        leaves = []
        if self.left_child:
            leaves.extend(self.left_child.get_leaves_below())
        if self.right_child:
            leaves.extend(self.right_child.get_leaves_below())
        return leaves


class Leaf(Node):
    """
    Represents a leaf node in a decision tree.

    Attributes:
        value (any): The value associated with the leaf node.
        depth (int): The depth of the leaf node in the decision tree.
    """

    def __init__(self, value, depth=None):
        """
        Initialize a DecisionTreeNode object.

        Args:
            value (any): The value associated with the node.
            depth (int, optional): The depth of the node in the decision tree.
            Defaults to None.
        """
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

    def count_nodes_below(self, only_leaves=False):
        """
        Counts the number of nodes below the leaf node.

        Args:
            only_leaves (bool): If True, counts only the leaf nodes.
            If False, counts all nodes.

        Returns:
            int: The number of nodes below the leaf node.
        """
        return 1

    def __str__(self):
        """
        Returns a string representation of the leaf node.

        Returns:
            str: The string representation of the leaf node.
        """
        return (f"-> leaf [value={self.value}] ")

    def get_leaves_below(self):
        """
        Returns a list of leaf nodes below the current leaf node.

        Returns:
            list: A list of leaf nodes below the current leaf node.
        """
        return [self]


class Decision_Tree():
    """
    Decision_Tree class represents a decision tree model.

    Attributes:
        max_depth (int): The maximum depth of the decision tree.
        Default is 10.
        min_pop (int): The minimum number of samples required to
        split a node. Default is 1.
        seed (int): The seed value for random number generation.
        Default is 0.
        split_criterion (str): The criterion used to split
        the nodes. Default is "random".
        root (Node): The root node of the decision tree
        If not provided, a new root node will be created.

    Methods:
        depth(): Returns the maximum depth of the decision tree.
        count_nodes(only_leaves=False): Returns the number of
        nodes in the decision tree. If only_leaves is True,
        it returns the number of leaf nodes only.
        __str__(): Returns a string representation of the decision tree.
        get_leaves(): Returns a list of leaf nodes in the decision tree.
    """

    def __init__(
            self,
            max_depth=10,
            min_pop=1,
            seed=0,
            split_criterion="random",
            root=None):
        """
        Initialize a DecisionTree object.

        Args:
            max_depth (int): The maximum depth of the decision tree.
            Defaults to 10.
            min_pop (int): The minimum number of samples required to
            split a node
            Defaults to 1.
            seed (int): The seed value for the random number generator
            Defaults to 0.
            split_criterion (str): The criterion used to split nodes
            Defaults to "random".
            root (Node): The root node of the decision tree. If not provided
            a new root node will be created.

        Attributes:
            rng (numpy.random.Generator): The random number generator.
            root (Node): The root node of the decision tree.
            explanatory (None): Placeholder for the explanatory variable.
            target (None): Placeholder for the target variable.
            max_depth (int): The maximum depth of the decision tree.
            min_pop (int): The minimum number of samples
            required to split a node.
            split_criterion (str): The criterion used to split nodes.
            predict (None): Placeholder for the predict function.
        """
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
        Calculate the depth of the decision tree.

        Returns:
            The depth of the decision tree.
        """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """
        Counts the number of nodes in the decision tree.

        Args:
            only_leaves (bool): If True, counts only the leaf nodes.
                                If False (default), counts all nodes.

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
        Returns a list of all the leaves below the root node.

        Returns:
            list: A list of all the leaves below the root node.
        """
        return self.root.get_leaves_below()
