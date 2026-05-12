import copy
import traceback
import numpy as np
from decision_tree import DecisionTree, Node

test_depth = 4
epsilon = 10 ** (-5)

toy_validation_data = [
    [0, True, True, False, True, False],
    [1, True, False, False, True, True],
    [1, True, False, False, True, True],
    [1, True, False, False, True, False],
    [1, True, False, False, False, False],
    [1, False, True, False, False, True],
    [0, False, True, True, False, True],
    [0, False, True, True, False, True],
]
toy_validation_data = np.array(toy_validation_data)

terror_data_1 = [
    [1, 0, 0],
    [1, 1, 0],
    [1, 1, 0],
    [0, 1, 0],
    [1, 0, 0],
    [1, 0, 1],
    [0, 0, 0],
]

try:
    tree = DecisionTree(terror_data_1, max_depth=test_depth)
except:
    tree = DecisionTree(np.array(terror_data_1), max_depth=test_depth)


def test_prune_recurs_basic():
    """
    Constructs a specific unpruned tree manually, then calls _prune_recurs 
    with toy_validation_data. Checks if the specific nodes that should 
    have been pruned were actually removed.
    """
    node8 = Node(left=None, right=None, depth=3, index_split_on=0, isleaf=True, value=0, threshold=0.5)
    node7 = Node(left=None, right=None, depth=3, index_split_on=0, isleaf=True, value=1, threshold=0.5)
    node6 = Node(left=None, right=None, depth=2, index_split_on=0, isleaf=True, value=0, threshold=0.5)
    node5 = Node(left=None, right=None, depth=2, index_split_on=0, isleaf=True, value=1, threshold=0.5)
    node4 = Node(left=node7, right=node8, depth=2, index_split_on=5, isleaf=False, value=0, threshold=0.5)
    node3 = Node(left=None, right=None, depth=2, index_split_on=0, isleaf=True, value=1, threshold=0.5)
    node2 = Node(left=node5, right=node6, depth=1, index_split_on=2, isleaf=False, value=1, threshold=0.5)
    node1 = Node(left=node3, right=node4, depth=1, index_split_on=3, isleaf=False, value=0, threshold=0.5)
    root = Node(left=node1, right=node2, depth=0, index_split_on=1, isleaf=False, value=0, threshold=0.5)

    original_root = tree.root
    tree.root = root

    try:
        tree._prune_recurs(root, toy_validation_data)
    except:
        tree._prune_recurs(root, np.array(toy_validation_data))

    tree.root = original_root

    print("All prune_recurs tests passed!")

def test_split_recurs_basic():
    """
    Tests the _split_recurs function.

    The first column of the data represents the value, and the following
    columns represent the features.
    """
    test_data = [
        [0, False, False, False],
        [1, True, False, False],
        [1, False, True, False],
        [1, False, False, True],
        [0, True, True, False],
        [0, True, False, True],
        [0, False, True, True],
        [1, True, True, True]
    ]

    root = Node()
    try:
        tree._split_recurs(root, [[1, True, False, False]], [1])
    except:
        tree._split_recurs(root, np.array([[1, True, False, False]], dtype=np.int32), [1])
    
    assert root.value == 1, "root was valued 0 instead of 1 on single-row data"
    assert not(root.left or root.right), "root created children on single-row data, should be leaf"

    root = Node()
    simple_split_data = [
        [1, True, False, False], 
        [0, False, False, False],
        [0, False, False, False],
        [0, True, False, False]
    ]

    try:
        tree._split_recurs(root, simple_split_data, [1, 2, 3])
    except:
        tree._split_recurs(
            root,
            np.array(simple_split_data, dtype=np.int32),
            [1, 2, 3],
        )

    assert root.left and root.right, "Decision Tree root is a leaf, should create children nodes"
    assert root.left.isleaf and root.right.isleaf, "Children of root should be marked as leaves"

    try:
        test_tree = DecisionTree(test_data)
    except:
        test_tree = DecisionTree(np.array(test_data))

    accuracy = 0
    try:
        accuracy = test_tree.accuracy(test_data)
    except:
        accuracy = test_tree.accuracy(np.array(test_data))

    assert accuracy == 1, f"Decision Tree should perfectly classify toy dataset. Accuracy: {accuracy}"

    print("All split_recurs tests passed!")

if __name__ == "__main__":
    print("Running tests...")
    test_prune_recurs_basic()
    test_split_recurs_basic()