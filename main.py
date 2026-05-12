from utils.get_data import *
from models.metrics import *
from models.decision_tree import *

# ###########################################################
# DATA PROCESSING
# ###########################################################


# ###########################################################
# DECISION TREE ALGORITHMS
# ###########################################################

# 1. Initialize
tree = DecisionTree(max_depth=5, min_samples=2)

# 2. Fit (assuming your class has a 'fit' wrapper for _split_recurs)
tree.fit(train_data)

# 3. Inspect
print("Initial Tree Structure:")
tree.print_tree()

# 4. Prune
tree._prune_recurs(tree.root, validation_data)

# 5. Evaluate
print(f"Final Accuracy: {tree.accuracy(test_data):.2%}")

# ###########################################################
# EVALUATION
# ###########################################################