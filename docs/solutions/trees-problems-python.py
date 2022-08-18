from typing import List, Optional


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def insertIntoBST(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
        """
        Insert into a Binary Search Tree - RECURSIVE
        Time: Worst - O(N), Avg - O(logN), Best - O(1)
        Space: Worst - O(N), Avg - O(logN), Best - O(1)
        """
        if not root:
            return TreeNode(val)
        if root.val < val:
            root.right = self.insertIntoBST(root.right, val)
        else:
            root.left = self.insertIntoBST(root.left, val)
        return root

    def insertIntoBST(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
        """
        Insert into a Binary Search Tree - ITERATIVE
        Time: Worst - O(N), Avg - O(logN), Best - O(1)
        Space: O(1)
        """
        parentNode = None
        curNode = root
        while curNode:
            parentNode = curNode
            if curNode.val > val:
                curNode = curNode.left
            else:
                curNode = curNode.right

        if not parentNode:
            return TreeNode(val)
        if parentNode.val > val:
            parentNode.left = TreeNode(val)
        else:
            parentNode.right = TreeNode(val)
        return root

    def deleteNode(self, root: Optional[TreeNode], key: int) -> Optional[TreeNode]:
        """
        Delete Node in a Binary Search Tree - RECURSIVE
        Time: Worst - O(N), Avg - O(logN), Best - O(1)
        Space: Worst - O(N), Avg - O(logN), Best - O(1)
        """
        if not root:
            return None
        if root.val == key:
            if not root.left:
                return root.right
            if not root.right:
                return root.left
            # move left sub-tree to most left of right sub-tree
            leftNode = TreeNode(root.left.val, root.left.left, root.left.right)
            curNode = root.right
            while curNode.left:
                curNode = curNode.left
            curNode.left = leftNode
            return root.right
        if root.val < key:
            root.right = self.deleteNode(root.right, key)
        else:
            root.left = self.deleteNode(root.left, key)
        return root

    def deleteNode(self, root: Optional[TreeNode], key: int) -> Optional[TreeNode]:
        """
        Delete Node in a Binary Search Tree - ITERATIVE
        Time: Worst - O(N), Avg - O(logN), Best - O(1)
        Space: O(1)
        """
        curNode = root
        parentNode = None
        while curNode and curNode.val != key:   # O(H)
            parentNode = curNode
            if curNode.val > key:
                curNode = curNode.left
            elif curNode.val < key:
                curNode = curNode.right

        if not curNode:
            return root

        if not curNode.left:
            subRoot = curNode.right
        elif not curNode.right:
            subRoot = curNode.left
        else:  # move left sub-tree to most left of right sub-tree
            leftNode = TreeNode(
                curNode.left.val, curNode.left.left, curNode.left.right)
            subRoot = rightNode = curNode.right
            while rightNode.left:    # O(H)
                rightNode = rightNode.left
            rightNode.left = leftNode

        if not parentNode:
            return subRoot
        if parentNode.val > key:
            parentNode.left = subRoot
        else:
            parentNode.right = subRoot
        return root

    def deleteNode(self, root: Optional[TreeNode], key: int) -> Optional[TreeNode]:
        """
        Another method of delete Node in a Binary Search Tree - RECURSIVE
        Time: Worst - O(N), Avg - O(logN), Best - O(1)
        Space: Worst - O(N), Avg - O(logN), Best - O(1)
        """
        if not root:
            return None
        if root.val == key:
            if not root.left:
                return root.right
            if not root.right:
                return root.left
            # find right most node on left sub-tree
            curNode = root.left
            while curNode.right:
                curNode = curNode.right
            root.left = self.deleteNode(root.left, curNode.val)
            root.val = curNode.val
            return root
        if root.val < key:
            root.right = self.deleteNode(root.right, key)
        else:
            root.left = self.deleteNode(root.left, key)
        return root

    def deleteNode(self, root: Optional[TreeNode], key: int) -> Optional[TreeNode]:
        """
        Another method of delete Node in a Binary Search Tree - ITERATIVE
        Time: Worst - O(N), Avg - O(logN), Best - O(1)
        Space: O(1)
        """
        curNode = root
        parentNode = None
        while curNode and curNode.val != key:  # O(H)
            parentNode = curNode
            if curNode.val > key:
                curNode = curNode.left
            elif curNode.val < key:
                curNode = curNode.right

        if not curNode:
            return root

        if not curNode.left:
            subRoot = curNode.right
        elif not curNode.right:
            subRoot = curNode.left
        else:  # find right most node on left sub-tree
            subRoot = curNode
            subParNode = None
            rightNode = curNode.left
            while rightNode.right:    # O(H)
                subParNode = rightNode
                rightNode = rightNode.right

            if subParNode:
                subParNode.right = rightNode.left
            else:
                subRoot.left = rightNode.left
            subRoot.val = rightNode.val

        if not parentNode:
            return subRoot
        if parentNode.val > key:
            parentNode.left = subRoot
        else:
            parentNode.right = subRoot
        return root

    def countNodes(self, root: Optional[TreeNode]) -> int:
        """
        Count Complete Tree Nodes - RECURSIVE
        Time: O(N)
        Space: O(N)
        """
        if not root:
            return 0
        return 1 + self.countNodes(root.left) + self.countNodes(root.right)

    def countNodes(self, root: Optional[TreeNode]) -> int:
        """
        Count Complete Tree Nodes - ITERATIVE
        Time: Worst - O(N), Best - O(H)
        Space: O(N)
        """
        if not root:
            return 0

        toCheck = []
        checked = set()
        level = 0
        totalNodes = 0
        curNode = root

        while curNode:   # O(H)
            totalNodes += pow(2, level)
            level += 1
            toCheck.append((curNode, level))
            curNode = curNode.right
        totalNodes += pow(2, level)

        # O(N) - to check last layer from right to left
        # I saw another solution that does Binary Search in last layer, then it's:
        # O(logN): Binary Search in O(N) of nodes in last layer
        #   *
        # O(logN): each traverse from root to last layer
        #   = O((logN)^2)
        while len(toCheck):
            popped, n = toCheck.pop()
            checked.add(popped)
            if popped.right in checked and popped.left in checked:
                continue
            if n < level:
                if popped.left not in checked:
                    toCheck.append((popped.left, n+1))
                if popped.right not in checked:
                    toCheck.append((popped.right, n+1))
                continue

            if popped.right:
                return totalNodes
            totalNodes -= 1
            if popped.left:
                return totalNodes
            totalNodes -= 1

        return totalNodes

    def maxDepth(self, root: Optional[TreeNode]) -> int:
        """
        Maximum Depth of Binary Tree - RECURSIVE
        Time: O(N)
        Space: O(N)
        """
        if not root:
            return 0
        return 1 + max(self.maxDepth(root.left), self.maxDepth(root.right))

    def maxDepth(self, root: Optional[TreeNode]) -> int:
        """
        Maximum Depth of Binary Tree - ITERATIVE
        Time: O(N)
        Space: O(N)
        """
        if not root:
            return 0

        levelNodes = [root]
        maxDep = 0

        while len(levelNodes):
            maxDep += 1
            childNodes = []
            for node in levelNodes:
                if node.left:
                    childNodes.append(node.left)
                if node.right:
                    childNodes.append(node.right)
            levelNodes = childNodes
        return maxDep

    def minValue(self, root: Optional[TreeNode]) -> int:
        """ (HAVEN'T RUN, NOT LEETCODE QUESTION)
        Find minimum data value in a non-empty Binary Search Tree - RECURSIVE
        Time: Worst - O(N), Avg - O(logN), Best - O(1)
        Space: Worst - O(N), Avg - O(logN), Best - O(1)
        """
        if not root.left:
            return root.val
        return self.minValue(root.left)

    def minValue(self, root: Optional[TreeNode]) -> int:
        """ (HAVEN'T RUN, NOT LEETCODE QUESTION)
        Find minimum data value in a non-empty Binary Search Tree - ITERATIVE
        Time: Worst - O(N), Avg - O(logN), Best - O(1)
        Space: O(1)
        """
        # ITERATIVE
        curNode = root
        while curNode.left:
            curNode = curNode.left
        return curNode.val

    def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
        """
        Find a root-to-leaf path that sums up to targetSum - RECURSIVE
        Time: O(N)
        Space: O(N)
        """
        if not root:
            return False
        if root.val == targetSum and not root.left and not root.right:
            return True
        remain = targetSum - root.val
        return self.hasPathSum(root.left, remain) or self.hasPathSum(root.right, remain)

    def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
        """
        Find a root-to-leaf path that sums up to targetSum - ITERATIVE
        Time: Avg - O(NlogN)
        Space: O(N)
        """
        # ITERATIVE
        if not root:
            return False
        rootToLeaf = []
        curSum = 0
        toCheck = [root]

        while len(toCheck):   # O(N)
            popped = toCheck.pop()

            while len(rootToLeaf):   # O(H)
                node = rootToLeaf[-1]
                if node.left != popped and node.right != popped:
                    delNode = rootToLeaf.pop()
                    curSum -= delNode.val
                else:
                    break

            rootToLeaf.append(popped)
            curSum += popped.val
            if popped.right:
                toCheck.append(popped.right)
            if popped.left:
                toCheck.append(popped.left)
            if not popped.left and not popped.right:
                if curSum == targetSum:
                    return True
        return False

    def binaryTreePaths(self, root: Optional[TreeNode]) -> List[str]:
        """
        Find all root-to-leaf paths in a Binary Tree - RECURSIVE
        Time: Worst - O(NlogN), Best - O(N)
        Space: Worst - O(NlogN), Best - O(N)
        """
        if not root:
            return []
        if not root.left and not root.right:
            return [f'{root.val}']
        paths = []
        if root.left:
            for path in self.binaryTreePaths(root.left):
                paths.append(f'{root.val}->{path}')
        if root.right:
            for path in self.binaryTreePaths(root.right):
                paths.append(f'{root.val}->{path}')
        return paths

    def binaryTreePaths(self, root: Optional[TreeNode]) -> List[str]:
        """
        Find all root-to-leaf paths in a Binary Tree - ITERATIVE
        Time: Worst - O(NlogN), Best - O(N)
        Space: Worst - O(NlogN), Best - O(N)
        """
        if not root:
            return []
        paths = []
        rootToLeaf = []
        toCheck = [root]

        while len(toCheck):    # O(N)
            popped = toCheck.pop()

            # O(H)
            while len(rootToLeaf) and rootToLeaf[-1].left != popped and rootToLeaf[-1].right != popped:
                rootToLeaf.pop()

            rootToLeaf.append(popped)
            if popped.left:
                toCheck.append(popped.left)
            if popped.right:
                toCheck.append(popped.right)
            if not popped.left and not popped.right:
                paths.append("->".join([str(n.val) for n in rootToLeaf]))
        return paths

    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        """
        Invert (mirror) a Binary Tree - RECURSIVE
        Time: O(N)
        Space: O(N)
        """
        if not root or (not root.left and not root.right):
            return root
        rightSubTree = self.invertTree(root.right)
        leftSubTree = self.invertTree(root.left)
        root.right = leftSubTree
        root.left = rightSubTree
        return root

    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        """
        Invert (mirror) a Binary Tree - ITERATIVE
        Time: O(N)
        Space: O(N)
        """
        if not root or (not root.left and not root.right):
            return root

        curLevelNodes = [root]
        while len(curLevelNodes):
            childNodes = []
            for node in curLevelNodes:
                left = node.left
                right = node.right
                if left:
                    childNodes.append(node.left)
                if right:
                    childNodes.append(node.right)
                node.left = right
                node.right = left
            curLevelNodes = childNodes

        return root

    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        """
        Check if two Binary Trees are the same - RECURSIVE
        Time: O(N)
        Space: O(N)
        """
        if not p and not q:
            return True
        if not p and q or p and not q:
            return False
        if p.val != q.val:
            return False
        return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)

    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        """
        Check if two Binary Trees are the same - ITERATIVE
        Time: O(N)
        Space: O(N)
        """
        # ITERATIVE
        if not p and not q:
            return True
        if not p and q or p and not q:
            return False
        levelNodesP = [p]
        levelNodesQ = [q]
        while len(levelNodesP) or len(levelNodesQ):
            if len(levelNodesP) != len(levelNodesQ):
                return False
            childNodesP = []
            childNodesQ = []
            for i in range(len(levelNodesP)):
                nodeP = levelNodesP[i]
                nodeQ = levelNodesQ[i]
                if not nodeP and not nodeQ:
                    continue
                if not nodeP or not nodeQ:
                    return False
                if nodeP.val != nodeQ.val:
                    return False
                childNodesP += [nodeP.left, nodeP.right]
                childNodesQ += [nodeQ.left, nodeQ.right]
            levelNodesP = childNodesP
            levelNodesQ = childNodesQ
        return True

    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        """
        Check if it is a valid binary search tree - RECURSIVE
        Time: O(N)
        Space: O(N)
        """
        # inorder then compare
        inOrderNodes = self.inOrder(root)
        if len(inOrderNodes) < 2:
            return True
        for i in range(len(inOrderNodes)-1):
            if inOrderNodes[i].val >= inOrderNodes[i+1].val:
                return False
        return True

    def inOrder(self, root: Optional[TreeNode]) -> List[TreeNode]:
        """
        Helper function of above isValidBST - RECURSIVE
        (ps: in discussion, there is also a way to create a helper function of isValidBSTLH(root, low, high))
        Time: O(N)
        Space: O(N)
        """
        if not root:
            return []
        return self.inOrder(root.left) + [root] + self.inOrder(root.right)

    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        """
        Check if it is a valid binary search tree - ITERATIVE
        Time: O(N)
        Space: O(N)
        """
        if not root:
            return True
        leftMostSoFar = root
        rightToLeft = []   # O(N)
        inOrder = []   # O(N)
        while leftMostSoFar or len(rightToLeft):
            while leftMostSoFar:
                rightToLeft.append(leftMostSoFar)
                leftMostSoFar = leftMostSoFar.left

            leftMost = rightToLeft.pop()
            if len(inOrder) and leftMost.val <= inOrder[-1].val:
                return False
            inOrder.append(leftMost)
            if leftMost.right:
                leftMostSoFar = leftMost.right
        # print([n.val for n in inOrder])
        return True

    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        """
        Right side view of a binary tree - RECURSIVE
        Time: O(N)
        Space: O(N)
        """
        if not root:
            return []
        if not root.left and not root.right:
            return [root.val]
        leftRSV = self.rightSideView(root.left)
        rightRSV = self.rightSideView(root.right)
        rsv = [root.val] + rightRSV
        if len(leftRSV) > len(rightRSV):
            rsv += leftRSV[len(rightRSV):]
        return rsv

    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        """
        Right side view of a binary tree - ITERATIVE
        Time: O(N)
        Space: O(N)
        """
        if not root:
            return []
        levelNodes = [root]
        rsv = []
        while len(levelNodes):
            numNodesInLevel = len(levelNodes)
            for i in range(numNodesInLevel):
                popped = levelNodes.pop()
                if popped.right:
                    levelNodes.insert(0, popped.right)
                if popped.left:
                    levelNodes.insert(0, popped.left)
                if i == 0:
                    rsv.append(popped.val)
        return rsv

    def convertBST(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        """
        Convert BST such that every key -> original key + sum of all keys > original key - RECURSIVE
        Time: O(N)
        Space: O(N)
        (ps: in discussion there is another helper function method, keep a global sum var in convertBST:
             convertBSTHelper(root.right), modify root val, convertBSTHelper(root.left))
        """
        if not root:
            return None
        rightToLeft = self.revInOrder(root)
        curSum = 0
        for node in rightToLeft:
            node.val += curSum
            curSum = node.val
        return root

    def revInOrder(self, root: Optional[TreeNode]) -> List[TreeNode]:
        """
        Helper function of reverse inOrder traversal
        Time: O(N)
        Space: O(N)
        """
        if not root:
            return []
        return self.revInOrder(root.right) + [root] + self.revInOrder(root.left)

    def convertBST(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        """
        Convert BST such that every key -> original key + sum of all keys > original key - ITERATIVE
        Time: O(N)
        Space: O(N)
        """
        if not root:
            return None

        rightMostSoFar = root
        leftToRight = []
        curSum = 0

        while rightMostSoFar or len(leftToRight):
            while rightMostSoFar:
                leftToRight.append(rightMostSoFar)
                rightMostSoFar = rightMostSoFar.right

            rightMost = leftToRight.pop()
            rightMost.val += curSum
            curSum = rightMost.val
            if rightMost.left:
                rightMostSoFar = rightMost.left

        return root

    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        """
        Lowest Common Ancestor of a Binary Search Tree - RECURSIVE
        Time: Worst - O(N), Avg - O(logN), Best - O(1)
        Space: Worst - O(N), Avg - O(logN), Best - O(1)
        """
        if not root:
            return None
        if p.val <= root.val and q.val >= root.val or p.val >= root.val and q.val <= root.val:
            return root
        if p.val < root.val and q.val < root.val:
            return self.lowestCommonAncestor(root.left, p, q)
        return self.lowestCommonAncestor(root.right, p, q)

    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        """
        Lowest Common Ancestor of a Binary Search Tree - ITERATIVE
        Time: O(N)
        Space: O(N)
        """
        if not root:
            return None
        levelNodes = [root]

        while (len(levelNodes)):
            numNodesInLevel = len(levelNodes)
            for i in range(numNodesInLevel):
                curNode = levelNodes.pop()
                if p.val <= curNode.val <= q.val or q.val <= curNode.val <= p.val:
                    return curNode
                if curNode.right:
                    levelNodes.insert(0, curNode.right)
                if curNode.left:
                    levelNodes.insert(0, curNode.left)

    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        """
        Lowest Common Ancestor of a Binary Search Tree - ITERATIVE
        (I saw a very good solution in discussion)
        Time: Worst - O(N), Avg - O(logN), Best - O(1)
        Space: O(1)
        """
        min_val, max_val = min(p.val, q.val), max(p.val, q.val)

        while True:
            if min_val <= root.val <= max_val:
                return root
            elif max_val < root.val:
                root = root.left
            else:
                root = root.right

    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        """
        Lowest Common Ancestor of a Binary Tree - RECURSIVE
        Time: O(N)
        Space: O(N)
        """
        if not root:
            return None
        if root.val == p.val or root.val == q.val:
            return root

        pPath = self.findPath(root, p)
        # print([n.val for n in pPath])
        qPath = self.findPath(root, q)
        # print([n.val for n in qPath])
        lowestAnc = root
        for i in range(min(len(pPath), len(qPath))):
            if pPath[i].val != qPath[i].val:
                break
            lowestAnc = pPath[i]
        return lowestAnc

    def findPath(self, root: 'TreeNode', targetNode: 'TreeNode') -> List[TreeNode]:
        """
        Helper function to find a path from root to targetNode - RECURSIVE
        Time: O(N)
        Space: O(N)
        """
        if not root or not targetNode:
            return []
        if root.val == targetNode.val:
            return [root]

        leftPath = self.findPath(root.left, targetNode)
        if len(leftPath):
            return [root] + leftPath
        rightPath = self.findPath(root.right, targetNode)
        if len(rightPath):
            return [root] + rightPath

        return []

    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        """
        Lowest Common Ancestor of a Binary Tree - ITERATIVE
        Time: O(N)
        Space: O(N)
        """
        if not root:
            return None
        if root.val == p.val or root.val == q.val:
            return root

        nodeParent = {root.val: None}   # O(N)
        rightToLeft = []    # O(N)
        leftMostSoFar = root
        targetPaths = []    # O(H)

        while leftMostSoFar or len(rightToLeft):   # DFS: O(N)
            while leftMostSoFar:
                rightToLeft.append(leftMostSoFar)
                if leftMostSoFar.left:
                    nodeParent[leftMostSoFar.left.val] = leftMostSoFar
                leftMostSoFar = leftMostSoFar.left

            popped = rightToLeft.pop()

            if popped.val in set({p.val, q.val}):
                path = set({popped})
                curNode = popped
                otherPath = None if not len(targetPaths) else targetPaths[0]
                # O(H) at most twice
                while curNode and curNode.val in nodeParent:
                    if otherPath and curNode in otherPath:
                        return curNode
                    curNode = nodeParent[curNode.val]
                    path.add(curNode)
                targetPaths.append(path)

            if popped.right:
                nodeParent[popped.right.val] = popped
                leftMostSoFar = popped.right

    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        """
        Lowest Common Ancestor of a Binary Tree - RECURSIVE
        (Another method seen in discussion section)
        Time: O(N)
        Space: O(N)
        """
        if not root:
            return None
        if root.val == p.val or root.val == q.val:
            return root

        leftAnc = self.lowestCommonAncestor(root.left, p, q)
        rightAnc = self.lowestCommonAncestor(root.right, p, q)
        if leftAnc and rightAnc:
            return root
        if leftAnc:
            return leftAnc
        return rightAnc
