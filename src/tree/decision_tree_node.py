class TreeNode:
    def __init__(self, feature_index=None, feature_name=None, threshold=None, left=None, right=None, value=None, is_leaf=False):
        self.feature_index = feature_index
        self.feature_name = feature_name
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.is_leaf = is_leaf

    