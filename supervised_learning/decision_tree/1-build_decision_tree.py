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
        is_leaf (bool): Indicates if this node is a leaf node.
        sub_population (None or int): The number of samples in
        the sub-population represented by this node.
        depth (int): The depth of this node in the tree.
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
        """
        Calculates the maximum depth of the tree below this node.

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
            only_leaves (bool): If True, counts only the leaf nodes.

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


class Leaf(Node):
    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        return 1


class Decision_Tree():
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
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        return self.root.count_nodes_below(only_leaves=only_leaves)
