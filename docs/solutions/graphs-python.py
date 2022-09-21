import heapq
import math
from typing import List, Set


class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        """
        Given beginWord and endWord, and a wordList, return the number of words in the shortest 
        transformation sequence from beginWord to endWord, or 0 if no such sequence exists.
        e.x. beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log","cog"] -> 5

        Time:  O(N * W^2) - N words in wordList, each word W chars
        Space: O(N * W)
        """
        all_words = set(wordList)  # time O(N), space O(N * W)
        # initial check
        if endWord not in all_words:
            return 0

        ladder_len = 0
        # set to record all the words that have been visited, space O(N * W)
        visited_words = set()
        # queue to do level search of graph made up with words, space O(N * W)
        level_words = [beginWord]

        # keep looking for words that are 1 char away from popped word, 1 level a batch
        while level_words:  # time O(N)
            ladder_len += 1
            for _ in range(len(level_words)):
                popped_word = level_words.pop(0)
                if popped_word == endWord:
                    return ladder_len

                # append all words that are 1 char away to level_words
                for i in range(len(popped_word)):  # O(W)
                    # change 1 char and see if this made-up-word is in all_words and not visited yet
                    made_up_word_chars = list(popped_word)
                    for ord_c in range(ord('a'), ord('z')+1):  # O(1) -> 26
                        made_up_word_chars[i] = chr(ord_c)
                        made_up_word = "".join(made_up_word_chars)  # O(W)
                        if made_up_word in all_words and made_up_word not in visited_words:
                            visited_words.add(made_up_word)
                            level_words.append(made_up_word)

        return 0

    def connectSticks(self, sticks: List[int]) -> int:
        """
        An array sticks, where sticks[i] is the length of the ith stick.
        You can connect any two sticks of lengths x and y into one stick by paying a cost of x + y.
        Return the minimum cost of connecting all the given sticks into one stick in this way.
        e.x. sticks = [2,4,3] -> 14
             Explanation: You start with sticks = [2,4,3].
             1. 2 + 3 = 5. Now you have sticks = [5,4].
             2. 5 + 4 = 9. Now you have sticks = [9].
             The total cost is 5 + 9 = 14.

        Time: O(NlogN)
        Space: O(1)
        """
        if len(sticks) == 1:
            return 0

        sticks.sort()  # O(NlogN)
        cost = 0
        while len(sticks) > 1:  # O(N)
            first_stick_len = sticks.pop(0)
            second_stick_len = sticks.pop(0)
            new_stick_len = first_stick_len + second_stick_len

            cost += new_stick_len
            # find a place to insert the new stick, O(logN)
            start, end = 0, len(sticks)-1
            while start <= end:
                mid = math.floor((start + end) / 2)
                if sticks[mid] < new_stick_len:
                    start = mid + 1
                else:
                    end = mid - 1
            sticks.insert(start, new_stick_len)

        return cost

    def connectSticks(self, sticks: List[int]) -> int:
        """
        (with priority queue)
        """
        if len(sticks) == 1:
            return 0

        heapq.heapify(sticks)
        cost = 0
        while len(sticks) > 1:
            first_stick_len = heapq.heappop(sticks)
            second_stick_len = heapq.heappop(sticks)
            new_stick_len = first_stick_len + second_stick_len

            cost += new_stick_len
            heapq.heappush(sticks, new_stick_len)

        return cost

    def numIslands(self, grid: List[List[str]]) -> int:
        """
        m x n 2D binary grid which represents a map of '1's (land) and '0's (water), 
        return the number of islands.
        An island is surrounded by water and formed by adjacent lands horizontally or vertically. 
        e.x. grid = [                          -> 1
                    ["1","1","1","1","0"],
                    ["1","1","0","1","0"],
                    ["1","1","0","0","0"],
                    ["0","0","0","0","0"]
                    ]

        Time: O(m * n)
        Space: O(1)
        """
        num_islands = 0

        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == "1":
                    num_islands += 1
                    self.dfs_mark_zero(grid, i, j)

        return num_islands

    def dfs_mark_zero(self, grid: List[List[str]], row: int, col: int):
        """
        Each land is examined for 4 times at max.
        """
        grid[row][col] = "0"
        if col < len(grid[0]) - 1 and grid[row][col+1] == "1":
            self.dfs_mark_zero(grid, row, col+1)
        if row < len(grid) - 1 and grid[row+1][col] == "1":
            self.dfs_mark_zero(grid, row+1, col)
        if col > 0 and grid[row][col-1] == "1":
            self.dfs_mark_zero(grid, row, col-1)
        if row > 0 and grid[row-1][col] == "1":
            self.dfs_mark_zero(grid, row-1, col)

    def solve(self, board: List[List[str]]) -> None:
        """
        m x n matrix board containing 'X' and 'O', capture all regions that are 4-directionally 
        surrounded by 'X'.
        A region is captured by flipping all 'O's into 'X's in that surrounded region.
        e.x. board = [["X","X","X","X"],   -> [["X","X","X","X"],
                      ["X","O","O","X"],       ["X","X","X","X"],
                      ["X","X","O","X"],       ["X","X","X","X"],
                      ["X","O","X","X"]]       ["X","O","X","X"]]

        Time: O(m * n)
        Space: O(m * n)
        """
        visited = [[False for _ in range(len(board[0]))]
                   for _ in range(len(board))]

        # doing dfs twice, once for checking if touching the board, second for flip
        for i in range(len(board)):
            for j in range(len(board[0])):
                if not visited[i][j] and board[i][j] == "O":
                    if not self.dfs_touch_boarder(board, visited, i, j, False):
                        self.dfs_flip(board, i, j, "O", "X")

    def dfs_touch_boarder(self, board: List[List[str]], visited: List[List[bool]], row: int, col: int, touched: bool) -> bool:
        visited[row][col] = True

        if row == 0 or row == len(board) - 1 or col == 0 or col == len(board[0]) - 1:
            touched = True

        if row > 0 and not visited[row-1][col] and board[row-1][col] == "O":
            above_touch = self.dfs_touch_boarder(
                board, visited, row-1, col, touched)
            if not touched:
                touched = above_touch

        if col > 0 and not visited[row][col-1] and board[row][col-1] == "O":
            left_touch = self.dfs_touch_boarder(
                board, visited, row, col-1, touched)
            if not touched:
                touched = left_touch

        if row < len(board) - 1 and not visited[row+1][col] and board[row+1][col] == "O":
            below_touch = self.dfs_touch_boarder(
                board, visited, row+1, col, touched)
            if not touched:
                touched = below_touch

        if col < len(board[0]) - 1 and not visited[row][col+1] and board[row][col+1] == "O":
            right_touch = self.dfs_touch_boarder(
                board, visited, row, col+1, touched)
            if not touched:
                touched = right_touch

        return touched

    def dfs_flip(self, board: List[List[str]], row: int, col: int, from_str: str, to_str: str):
        board[row][col] = to_str

        if col > 0 and board[row][col-1] == from_str:
            self.dfs_flip(board, row, col-1, from_str, to_str)
        if row > 0 and board[row-1][col] == from_str:
            self.dfs_flip(board, row-1, col, from_str, to_str)
        if col < len(board[0]) - 1 and board[row][col+1] == from_str:
            self.dfs_flip(board, row, col+1, from_str, to_str)
        if row < len(board) - 1 and board[row+1][col] == from_str:
            self.dfs_flip(board, row+1, col, from_str, to_str)

    def solve(self, board: List[List[str]]) -> None:
        """
        (another algorithm to find all the regions touching boarders first)

        Time: O(m * n)
        Space: O(1)
        """
        for col in range(len(board[0])):
            if board[0][col] == "O":
                self.dfs_flip(board, 0, col, "O", "B")
            if board[-1][col] == "O":
                self.dfs_flip(board, len(board)-1, col, "O", "B")

        for row in range(len(board)):
            if board[row][0] == "O":
                self.dfs_flip(board, row, 0, "O", "B")
            if board[row][-1] == "O":
                self.dfs_flip(board, row, len(board[0])-1, "O", "B")

        for row in range(len(board)):
            for col in range(len(board[0])):
                if board[row][col] == "B":
                    board[row][col] = "O"
                else:
                    board[row][col] = "X"


class Graph:
    num_nodes = 0
    edges = {}

    dfs_num = 0

    def formGraph(self, n, connections, directed=False):
        self.num_nodes = n
        self.edges = {}
        for node in range(n):
            self.edges[node] = []
        for edge in connections:
            self.edges[edge[0]].append(edge[1])
            if not directed:
                self.edges[edge[1]].append(edge[0])

    def criticalConnections(self, n: int, connections: List[List[int]]) -> List[List[int]]:
        """
        (articulate bridge problem)
        n servers from 0 to n - 1 connected by undirected connections and connections[i] = [ai, bi] 
        represents a connection between servers ai and bi. 
        A critical connection is a connection that, if removed, will make some servers disconnected.
        Return all critical connections in the network in any order.
        e.x. Input: n = 4, connections = [[0,1],[1,2],[2,0],[1,3]] -> [[1,3]]

        Time: O(n + len(conn))
        Space: O(n) - not considering output array and graph obj
        """
        self.formGraph(n, connections)
        # print(self.edges)
        # order of DFS traversal of the graph, at which round it is visited
        dfs_nums = [-1] * n
        # dfs_num of some node that is cloeset to root node which the current node can reach
        # either through it's backedge or it's children's backedge
        low_nums = [-1] * n

        critical_edges = []
        self.dfs_articulate(0, -1, dfs_nums, low_nums, critical_edges)
        return critical_edges

    def dfs_articulate(self, node: int, parent: int, dfs_nums: List[int], low_nums: List[int], critical_edges: List[List[int]]):
        self.dfs_num += 1
        dfs_nums[node] = self.dfs_num
        low_nums[node] = self.dfs_num

        for child in self.edges[node]:
            if dfs_nums[child] == -1:
                self.dfs_articulate(child, node, dfs_nums,
                                    low_nums, critical_edges)
                # update my low to the lowest low of my children's
                # if my children can reach that node, I can too
                low_nums[node] = min(low_nums[node], low_nums[child])
                # if my child cannot reach me or some of my parents through backedges, I am a critical point
                if low_nums[child] > dfs_nums[node]:
                    critical_edges.append([node, child])
            elif parent != -1 and child != parent:  # a backedge
                # update my low to the dfs_num of the node this backedge reaches if it's closer to root
                low_nums[node] = min(low_nums[node], dfs_nums[child])

        # print(dfs_nums, low_nums)

    def validTree(self, n: int, edges: List[List[int]]) -> bool:
        """
        Check if the graph makes a valid tree.

        Time: O(n + e)
        Space: O(n) - not considering graph obj
        """
        if len(edges) != n-1:
            return False

        self.formGraph(n, edges)

        root_to_leaf = set()
        nodes_to_visit = set([node for node in range(n)])
        has_cycle = self.dfs_check_backedge(
            0, -1, root_to_leaf, nodes_to_visit)

        return not has_cycle and not nodes_to_visit

    def dfs_check_backedge(self, node: int, parent: int, root_to_leaf: Set[int], nodes_to_visit: Set[int]) -> bool:
        # print(node, parent, root_to_leaf)
        root_to_leaf.add(node)
        nodes_to_visit.remove(node)

        for child in self.edges[node]:
            if child == parent:
                continue
            if child in root_to_leaf:
                return True
            if child in nodes_to_visit and self.dfs_check_backedge(child, node, root_to_leaf, nodes_to_visit):
                return True

        root_to_leaf.remove(node)
        return False

    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        """
        There are a total of numCourses courses you have to take, labeled from 0 to numCourses - 1. 
        prerequisites[i] = [ai, bi] indicates that you must take course bi before ai.
        Check if can finish all courses.
        e.x. numCourses = 2, prerequisites = [[1,0],[0,1]] -> false
             Explanation: To take course 1 you should have finished course 0, and to take course 0 
                          you should also have finished course 1. So it is impossible.

        Time: O(n + p)
        Space: O(n) - not considering graph obj
        """
        if not prerequisites:
            return True

        self.formGraph(numCourses, prerequisites, directed=True)
        # print(self.edges)

        nodes_to_visit = set([n for n in range(numCourses) if self.edges[n]])
        root_to_leaf = set()
        while nodes_to_visit:
            root_node = next(iter(nodes_to_visit))
            if self.dfs_check_backedge_directed(root_node, root_to_leaf, nodes_to_visit):
                return False

        return True

    def dfs_check_backedge_directed(self, node: int, root_to_leaf: Set[int], nodes_to_visit: Set[int]) -> bool:
        # print(node, root_to_leaf)
        root_to_leaf.add(node)
        nodes_to_visit.remove(node)

        for child in self.edges[node]:
            # detect a backedge from my child me and my parents
            if child in root_to_leaf:
                return True
            # detect a backedge within my subtree
            if child in nodes_to_visit and self.dfs_check_backedge_directed(child, root_to_leaf, nodes_to_visit):
                return True

        # pop node to move to another path
        root_to_leaf.remove(node)
        return False
