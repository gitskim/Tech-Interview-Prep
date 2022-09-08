from typing import List, Optional


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    """
    PathSum III - find the number of paths where the sum of the values along the path equals targetSum
                  traveling only from parent nodes to child nodes
    """

    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> int:
        """
        RECURSIVE

        Time: O(N^2)
        Space: O(H)
        """
        if not root:
            return 0

        root_sum = self.pathSumFromRoot(root, targetSum)

        if root.left:
            root_sum += self.pathSum(root.left, targetSum)
        if root.right:
            root_sum += self.pathSum(root.right, targetSum)

        return root_sum

    def pathSumFromRoot(self, root: Optional[TreeNode], targetSum: int) -> int:
        if not root:
            return 0

        p_sum_r = 1 if root.val == targetSum else 0

        if root.left:
            p_sum_r += self.pathSumFromRoot(root.left, targetSum-root.val)
        if root.right:
            p_sum_r += self.pathSumFromRoot(root.right, targetSum-root.val)

        return p_sum_r

    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> int:
        """
        ITERATIVE with prefix (algorithm from solution)

        Time: O(N)
        Space: O(H)
        """
        # when no root or only 1 root
        if not root:
            return 0
        if not root.left and not root.right:
            if root.val == targetSum:
                return 1
            return 0

        num_valid_paths = 0

        # keep track of every node on a path's sum to root (sum of root - ... - each node on a path)
        # so if we later encounter a node, its path sum to root - some other node's path sum to root,
        # we have a subpath's sum == targetSum
        # e.x. 1 - 2 - 3 - 4 as a root to leaf path, each node's sum to root is {1: 1, 2: 3, 3: 6, 4: 10}
        #      if targetSum == 3, then [1 - 2] is a valid subpath because 2's sum to root == 3
        #      also [3] is a valid subpath because 3's sum to root (6) - 2's sum to root (3) == 3
        # but we only need to know how many subpaths are valid, not which ones, so we only need to keep
        # track of number of occurances of each sum, instead of each sum belong to which node
        # e.x. 1 - 2 - 3 - 4, targetSum == 3, sums count is {1: 1, 3: 1, 6: 1, 10: 1},
        #      1 node's sum to root is 1, 1 node's sum to root is 3, ...
        #      so we know, 1 path is valid because it's sum to root == 3,
        #      1 more path is valid because it's sum to root (6) - some 1 sum to root (3) == 3
        sum_to_root_count_on_path = {}  # space O(H)
        sum_to_root = 0
        # keep track of a certain path's nodes - space O(H)
        root_to_leaf = []
        # we still keep track of each node's sum so we can modify sum_count_on_path easily when trim nodes in root_to_leaf
        node_sum_to_root = {}  # space O(H)

        # in-order traverse (DFS) for each path - O(N)
        in_order = [root]
        while in_order:
            popped = in_order.pop()
            # trim root_to_leaf if switched to a new path, and update sum records
            while root_to_leaf:
                last_node = root_to_leaf[-1]
                if last_node.left == popped or last_node.right == popped:
                    break
                root_to_leaf.pop()
                sum_to_root -= last_node.val
                sum_to_del = node_sum_to_root.pop(last_node)
                sum_to_root_count_on_path[sum_to_del] -= 1
                if sum_to_root_count_on_path[sum_to_del] == 0:
                    sum_to_root_count_on_path.pop(sum_to_del)

            # keep going with traverse and update sum records
            root_to_leaf.append(popped)
            if popped.right:
                in_order.append(popped.right)
            if popped.left:
                in_order.append(popped.left)

            sum_to_root += popped.val
            # check if sum_to_root for this node == targetSum
            if sum_to_root == targetSum:
                num_valid_paths += 1
            # check if sum_to_root for this node - sum_to_root for some other node on this path == targetSum
            # num of other nodes' sum that fit this requirement is num of valid paths
            sum_to_root_to_find = sum_to_root - targetSum
            if sum_to_root_to_find in sum_to_root_count_on_path:
                num_valid_paths += sum_to_root_count_on_path[sum_to_root_to_find]
            # record this sum_to_root as well
            node_sum_to_root[popped] = sum_to_root
            if sum_to_root not in sum_to_root_count_on_path:
                sum_to_root_count_on_path[sum_to_root] = 0
            sum_to_root_count_on_path[sum_to_root] += 1

        return num_valid_paths

    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> int:
        """
        RECURSIVE with prefix (algorithm from solution)
        It's the same idea as iterative, DFS and keep count of sum_to_root, but since it's recursive
        we need to pass these records to child nodes in calls.

        Time: O(N)
        Space: O(H)
        """
        # when no root or only 1 root
        if not root:
            return 0
        if not root.left and not root.right:
            if root.val == targetSum:
                return 1
            return 0

        return self.pathSumDFS(root, targetSum, 0, {})

    def pathSumDFS(self, root: Optional[TreeNode], targetSum: int, sum_parent_to_root: int, sum_to_root_count_on_path: dict[int, int]):
        # check valid paths ending with root
        sum_to_root = sum_parent_to_root + root.val
        num_valid_paths = 0
        if sum_to_root == targetSum:
            num_valid_paths += 1
        sum_to_find = sum_to_root - targetSum
        if sum_to_find in sum_to_root_count_on_path:
            num_valid_paths += sum_to_root_count_on_path[sum_to_find]

        # form a new count dict to pass onto child nodes (python pass dict by reference)
        sum_to_root_count_cp = sum_to_root_count_on_path.copy()
        if sum_to_root not in sum_to_root_count_cp:
            sum_to_root_count_cp[sum_to_root] = 0
        sum_to_root_count_cp[sum_to_root] += 1

        # keep checking on child nodes
        left_valid_paths = right_valid_paths = 0
        if root.left:
            left_valid_paths = self.pathSumDFS(
                root.left, targetSum, sum_to_root, sum_to_root_count_cp)
        if root.right:
            right_valid_paths = self.pathSumDFS(
                root.right, targetSum, sum_to_root, sum_to_root_count_cp)

        return num_valid_paths + left_valid_paths + right_valid_paths


class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        """
        (RECURSIVE)
        Given an integer array nums of unique elements, return all possible subsets (the power set).

        Time: O(2^N)
        Space: O(2^N * N) - 2^N subsets, each subset has 0~N numbers (avg N/2 numbers per subset)
               O(2^N) if not counting return output
        """
        if not nums:
            return [[]]

        prev_subsets = self.subsets(nums[:-1])

        return prev_subsets + [ps + [nums[-1]] for ps in prev_subsets]

    def subsets(self, nums: List[int]) -> List[List[int]]:
        """
        (ITERATIVE)

        Time: O(2^N)
        Space: O(2^N * N)
               O(1) if not counting return output
        """
        all_subsets = [[]]

        # 2^0 + 2^1 + ... + 2^(N-1) = 2^N - 1
        for num in nums:
            all_subsets += [s + [num] for s in all_subsets]

        return all_subsets

    def subsets(self, nums: List[int]) -> List[List[int]]:
        """
        (BACKTRACK)

        Time: O(2^N)
        Space: O(2^N * N)
        """
        answer = [[]]
        for i in range(len(nums)):  # O(N)
            self.backtrack(nums, [], i, answer)
        return answer

    def backtrack(self, nums: List[int], prefix: List[int], startI: int, allSubsets: List[List[int]]):
        allSubsets.append(prefix + [nums[startI]])
        for i in range(startI + 1, len(nums)):
            self.backtrack(nums, prefix + [nums[startI]], i, allSubsets)


class Solution:
    """
    The diameter of a binary tree is the length of the longest path between any two nodes in a tree. 
    This path may or may not pass through the root.
    """
    diameter = 0

    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        """RECURSIVE"""
        self.height_and_update_diameter(root)
        return self.diameter

    def height_and_update_diameter(self, root: Optional[TreeNode]) -> int:
        """
        Time: O(N)
        Space: O(H)
        """
        if not root or not root.left and not root.right:
            return 0

        cur_diameter = 0
        left_height = self.height_and_update_diameter(root.left)
        right_height = self.height_and_update_diameter(root.right)
        if root.left:
            cur_diameter += left_height + 1
        if root.right:
            cur_diameter += right_height + 1
        self.diameter = max(self.diameter, cur_diameter)

        return max(left_height, right_height) + 1

    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        """
        ITERATIVE

        Time: O(N)
        Space: O(N)
        """
        if not root:
            return 0

        diameter = 0
        # height of each node - space O(N)
        node_height = {}
        # keep track of the current path from root to leaf - space O(H)
        root_to_leaf = []

        # do an in_order traverse (DFS) of the tree
        in_order = [root]
        while in_order:  # O(N)
            popped = in_order.pop()
            # currently height 0 without knowing any other info
            node_height[popped] = 0

            # if we are looking at a node that's on a new path, we need to trim root_to_leaf until we see
            # its parent, and we update the height info during the process
            while root_to_leaf:  # O(N) in total
                last_node = root_to_leaf[-1]
                # stop until we find popped's parent
                if last_node.left == popped or last_node.right == popped:
                    break

                # trim root_to_leaf
                root_to_leaf.pop()
                last_node_parent = root_to_leaf[-1]

                # whenever we pop a node from root_to_leaf, update max diameter for its parent if needed
                # we pop each node from root_to_leaf only once, so for a certain parent node:
                #  - when left_child popped, parent_height == 0, diameter update to left_child_height + 1
                #  - when right_child popped, parent_height == left_child_height + 1, diameter update to full diameter at this parent node
                # and then compare with max diameter
                diameter = max(
                    diameter, node_height[last_node_parent] + node_height[last_node] + 1)

                # whenever we pop a node from root_to_leaf, update its parent's height if needed
                # we pop each node from root_to_leaf only once, so for a certain parent node:
                #  - when left_child popped, parent_height == 0, parent_height update to left_child_height + 1
                #  - when right_child popped, parent_height == left_child_height + 1, parent_height update to max(left_height+1, right_height+1)
                node_height[last_node_parent] = max(
                    node_height[last_node_parent], node_height[last_node] + 1)

            # keep going with in_order (DFS)
            root_to_leaf.append(popped)

            if popped.right:
                in_order.append(popped.right)
            if popped.left:
                in_order.append(popped.left)

        # traverse done, root_to_leaf has one path left (the right-most one), trim to root and update height info as well
        while len(root_to_leaf) > 1:
            last_node = root_to_leaf.pop()
            last_node_parent = root_to_leaf[-1]
            diameter = max(
                diameter, node_height[last_node_parent] + node_height[last_node] + 1)
            node_height[last_node_parent] = max(
                node_height[last_node_parent], node_height[last_node] + 1)
        return diameter


class Solution:
    """
    Given the root of a binary tree, return the maximum path sum of any non-empty path.
    """
    max_sum = None

    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        """
        RECURSIVE

        Time: O(N)
        Space: O(H)
        """
        self.maxSinglePathSumFromRoot(root)
        return self.max_sum

    def maxSinglePathSumFromRoot(self, root: Optional[TreeNode]) -> int:
        """
        Helper function to get max sum of a single side path starting from root, and update max_sum if
        a path sum containing root is > max_sum.
        e.x.  2
            3   4
            return 6 (2+4)
            update max_sum to 9 (3+2+4)

        Time: O(N)
        Space: O(H)
        """
        max_single_path_sum = root.val
        left_max_single = right_max_single = None

        # get max sum of single paths starting from root.left and root.right
        if root.left:
            left_max_single = self.maxSinglePathSumFromRoot(root.left)
        if root.right:
            right_max_single = self.maxSinglePathSumFromRoot(root.right)

        # update max_sum if needed
        max_path_sum_root = root.val

        if left_max_single and left_max_single > 0:
            max_path_sum_root += left_max_single
            # compare left and right max single path's sum and decide root max single path's sum
            if not right_max_single or left_max_single >= right_max_single:
                max_single_path_sum += left_max_single
        if right_max_single and right_max_single > 0:
            max_path_sum_root += right_max_single
            # compare left and right max single path's sum and decide root max single path's sum
            if not left_max_single or right_max_single > left_max_single:
                max_single_path_sum += right_max_single

        if not self.max_sum or max_path_sum_root > self.max_sum:
            self.max_sum = max_path_sum_root

        return max_single_path_sum

    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        """
        ITERATIVE

        Time: O(N)
        Space: O(N)
        """
        max_sum = root.val

        # keep max sum of a single path start from each node - space O(N)
        node_max_single_path_sum = {}
        # current root to leaf path nodes while traversing - space O(H)
        root_to_leaf = []

        # start in_order traverse (DFS) - O(N)
        in_order = [root]
        while in_order:
            popped = in_order.pop()
            node_max_single_path_sum[popped] = popped.val
            max_sum = max(max_sum, popped.val)

            # if we pop a node that's on a different path, need to trim root_to_leaf and update sum
            while root_to_leaf:  # O(N) in total
                last_node = root_to_leaf[-1]
                if last_node.left == popped or last_node.right == popped:
                    break
                root_to_leaf.pop()
                last_node_parent = root_to_leaf[-1]
                max_sum = max(
                    max_sum,
                    node_max_single_path_sum[last_node_parent] +
                    node_max_single_path_sum[last_node]
                )
                node_max_single_path_sum[last_node_parent] = max(
                    node_max_single_path_sum[last_node_parent],
                    node_max_single_path_sum[last_node] + last_node_parent.val
                )

            # go ahead with the traverse
            root_to_leaf.append(popped)
            if popped.right:
                in_order.append(popped.right)
            if popped.left:
                in_order.append(popped.left)

        # one right-most path left in root_to_leaf, finish checking
        while len(root_to_leaf) > 1:
            last_node = root_to_leaf.pop()
            last_node_parent = root_to_leaf[-1]
            max_sum = max(
                max_sum,
                node_max_single_path_sum[last_node_parent] +
                node_max_single_path_sum[last_node]
            )
            node_max_single_path_sum[last_node_parent] = max(
                node_max_single_path_sum[last_node_parent],
                node_max_single_path_sum[last_node] + last_node_parent.val
            )

        return max_sum
