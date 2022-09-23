import heapq
import math
from typing import Dict, List, Set


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

    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        """
        (Topological ordering problem)
        There are a total of numCourses courses you have to take, labeled from 0 to numCourses - 1. 
        prerequisites[i] = [ai, bi] indicates that you must take course bi before ai.
        Find the ordering of courses you should take to finish all courses.
        e.x. numCourses = 4, prerequisites = [[1,0],[2,0],[3,1],[3,2]] -> [0,2,1,3]

        Time: O(n + p)
        Space: O(n)
        """
        if not prerequisites:
            return [c for c in range(numCourses)]

        # reverse each edge to make edges of [ai, bi] such that you must take ai before bi
        before_afters = [[bi, ai] for [ai, bi] in prerequisites]
        self.formGraph(numCourses, before_afters, directed=True)

        # dfs and form the reverse order
        reverse_order = []
        root_to_leaf = set()
        nodes_to_visit = set([c for c in range(numCourses)])

        for c in range(numCourses):
            if c in nodes_to_visit:
                if not self.dfs_reverse(c, reverse_order, root_to_leaf, nodes_to_visit):
                    return []

        # reverse and return the correct order
        return list(reversed(reverse_order))

    def dfs_reverse(self, node: int, reverse_order: List[int], root_to_leaf: Set[int], nodes_to_visit: Set[int]) -> bool:
        """
        Time: O(V+E)
        Space: O(V)
        """
        root_to_leaf.add(node)
        nodes_to_visit.remove(node)

        if node in self.edges:
            for child in self.edges[node]:
                # detect a cycle
                if child in root_to_leaf:
                    return False
                if child in nodes_to_visit:
                    if not self.dfs_reverse(child, reverse_order, root_to_leaf, nodes_to_visit):
                        return False

        reverse_order.append(node)
        root_to_leaf.remove(node)
        return True

    def accountsMerge(self, accounts: List[List[str]]) -> List[List[str]]:
        """
        Merge email accounts of a same person. If two account lists share a same email, they belong to one.
        Accounts can be in any order, emails within an account need to be sorted.
        e.x. accounts = [["John","johnsmith@mail.com","john_newyork@mail.com"],
                         ["John","johnsmith@mail.com","john00@mail.com"],
                         ["Mary","mary@mail.com"],
                         ["John","johnnybravo@mail.com"]]
                     -> [["John","john00@mail.com","john_newyork@mail.com","johnsmith@mail.com"],
                         ["Mary","mary@mail.com"],
                         ["John","johnnybravo@mail.com"]]
             Explanation: 1st and 2nd have the common email "johnsmith@mail.com".
                          3rd John and Mary are different people as none of their email addresses are 
                          used by other accounts.

        Time: O(N^2 * A + N * M_AlogM_A), A: account len, M_A: merged account len
        Space: O(N + E) < O(N^2), E: number of edges, connected accounts
        """
        # form an undirected graph that connects accounts of a same person together
        # each edge means two accounts belong to same person (share an email)
        # O(N^2 * max account len)
        edges = [set() for _ in range(len(accounts))]
        for i, ac in enumerate(accounts):  # O(N)
            for j in range(i+1, len(accounts)):  # O(N)
                common_emails = set(ac[1:]).intersection(
                    set(accounts[j][1:]))  # O(max account len)
                if common_emails:
                    edges[i].add(j)
                    edges[j].add(i)

        # print(edges)

        # find connected parts in graph, with dfs
        nodes_to_visit = set([a for a in range(len(accounts))])  # space O(V)
        merged_accounts = {}  # space O(E)
        while nodes_to_visit:  # time O(V+E) < O(N^2)
            next_root = next(iter(nodes_to_visit))
            connected_nodes = set()
            self.dfs_connected(next_root, connected_nodes,
                               nodes_to_visit, edges)
            merged_accounts[next_root] = connected_nodes

        # print(merged_accounts)
        merged_ac_emails = []
        for name_id, ac_ids in merged_accounts.items():
            ac_emails = set()
            for id in ac_ids:
                # O(len of max account -> A)
                ac_emails = ac_emails.union(set(accounts[id][1:]))
            merged_ac_emails.append(
                [accounts[name_id][0]] + sorted(list(ac_emails)))  # O(M_AlogM_A) - M_A: len of merged account
        return merged_ac_emails

    def dfs_connected(self, node: int, connected_nodes: Set[int], nodes_to_visit: Set[int], edges: List[Set[int]]):
        nodes_to_visit.remove(node)
        connected_nodes.add(node)

        for child in edges[node]:
            if child in nodes_to_visit:
                self.dfs_connected(child, connected_nodes,
                                   nodes_to_visit, edges)

    def accountsMerge(self, accounts: List[List[str]]) -> List[List[str]]:
        """
        (another brute force algo)
        Time: O(N^2 + ElogE) - N: num of accounts, E: num of total emails
        Space: O(N + E)
        """
        account_owners = {}  # space O(N)
        account_emails = {}  # space O(N + E)
        for i, ac in enumerate(accounts):  # O(N)
            account_owners[i] = i
            account_emails[i] = set(ac[1:])  # O(E) in total

            # check if overlap with any earlier accounts
            for j in range(i):  # O(N)
                if account_emails[j].intersection(account_emails[i]):
                    account_owners[i] = account_owners[j]
                    account_emails[j] = account_emails[j].union(
                        account_emails[i])
                    break

            # update all accounts connected to me as well
            if account_owners[i] < i:
                for j in range(i):  # O(N)
                    if account_owners[j] != account_owners[i] and account_emails[j].intersection(account_emails[i]):
                        account_owners[j] = account_owners[i]
                        account_emails[account_owners[i]] = account_emails[account_owners[i]].union(
                            account_emails[j])
                        account_emails[j] = set()

                account_emails[i] = set()

        # generate merged accounts, O(N + ElogE)
        merged_accounts = []
        for ac_id, ac_emails in account_emails.items():
            if ac_emails:
                merged_accounts.append(
                    [accounts[ac_id][0]] + sorted(list(ac_emails)))

        return merged_accounts

    def accountsMerge(self, accounts: List[List[str]]) -> List[List[str]]:
        """
        (another dfs algo from solution - node as single email, way faster)
        Time: O(ElogE) - E: num of total emails
        Space: O(E)
        """
        edges = {}  # space O(E)
        emails_to_visit = set()  # space O(E)

        # form a graph where nodes are connected if they are shared by the same person
        # time O(E) in total
        for ac in accounts:
            if ac[1] not in edges:
                edges[ac[1]] = set()
            emails_to_visit.add(ac[1])

            if len(ac) == 2:
                continue
            for i in range(2, len(ac)):
                if ac[i] == ac[i-1]:
                    continue
                if ac[i] not in edges:
                    edges[ac[i]] = set()
                edges[ac[i-1]].add(ac[i])
                emails_to_visit.add(ac[i])

        # make all edges double direction
        for node, childs in edges.items():
            for child in childs:
                if child not in edges:
                    edges[child] = set()
                edges[child].add(node)

        # print(edges)

        merged_accounts = []
        for ac in accounts:
            if ac[1] in emails_to_visit:
                connected_emails = set()
                self.dfs_emails(ac[1], connected_emails,
                                emails_to_visit, edges)
                merged_accounts.append(
                    [ac[0]] + sorted(list(connected_emails)))

        return merged_accounts

    def dfs_emails(self, email: str, connected_emails: List[str], emails_to_visit: Set[str], edges: Dict[str, Set[str]]):
        connected_emails.add(email)
        emails_to_visit.remove(email)

        for child in edges[email]:
            if child in emails_to_visit:
                self.dfs_emails(child, connected_emails,
                                emails_to_visit, edges)

    def accountsMerge(self, accounts: List[List[str]]) -> List[List[str]]:
        """
        (another disjoint union set algo from solution - node as single email, faster than earlier DUS algo)
        Time: O(N^2 * E + ElogE) - N: num of accounts, E: num of total emails
        Space: O(N + E)
        """
        group_parent = {a: a for a in range(
            len(accounts))}  # time O(N), space O(N)
        # time O(N), space O(N)
        group_rank = {a: 0 for a in range(len(accounts))}
        email_group = {}  # space O(E)

        # time O(N)
        def find_parent(group_id: int) -> int:
            while group_parent[group_id] != group_id:
                group_id = group_parent[group_id]
            return group_id

        # time O(N)
        def merge_group_id(group_id_1: int, group_id_2: int):
            parent_1 = find_parent(group_id_1)
            parent_2 = find_parent(group_id_2)

            if group_rank[parent_1] > group_rank[parent_2]:
                group_parent[parent_2] = parent_1
            elif group_rank[parent_1] < group_rank[parent_2]:
                group_parent[parent_1] = parent_2
            else:
                group_parent[parent_2] = parent_1
                group_rank[parent_1] += 1

        # time O(N)
        for group_id, ac in enumerate(accounts):
            for email in ac[1:]:  # time O(E) in total
                if email not in email_group:
                    email_group[email] = group_id
                else:
                    merge_group_id(email_group[email], group_id)  # time O(N)

        # print(f"group parent: {group_parent}")
        # print(f"group rank: {group_rank}")
        # print(f"email group: {email_group}")

        merged_id_emails = {}
        for email, group_id in email_group.items():  # time O(E)
            # find root of the same group
            parent = find_parent(group_id)  # time O(N)
            # add email to group
            if parent not in merged_id_emails:
                merged_id_emails[parent] = set()
            merged_id_emails[parent].add(email)

        # time O(ElogE + N)
        return [[accounts[group_id][0]] + sorted(list(emails)) for group_id, emails in merged_id_emails.items()]

    def findMinHeightTrees(self, n: int, edges: List[List[int]]) -> List[int]:
        """
        Find the list of roots of all minimum height trees (MHTs) formed by edges.
        e.x. n = 4, edges = [[1,0],[1,2],[1,3]] -> [1]
             all trees formed by edges: 0 - 1 - 2/3
                                        1 - 0/2/3
                                        2 - 1 - 0/3
                                        3 - 1 - 0/2

        Time: O(N^2) - since it's a tree, it has (N-1) edges: O(N * (N + E)) -> O(N^2)
        Space: O(N)
        """
        min_height = n
        mht_roots = []

        self.formGraph(n, edges)
        all_nodes = set([x for x in range(n)])
        # try DFS on each root node, and see if we can beat the current min_height
        all_roots_to_check = all_nodes.copy()
        while all_roots_to_check:
            root = next(iter(all_roots_to_check))
            root_to_leaf = []
            nodes_to_visit = all_nodes.copy()
            root_height = self.dfs_height(
                root, root_to_leaf, nodes_to_visit, min_height)
            # print(root, root_height)
            if root_height == -1 or root_height > min_height:
                continue
            if root_height == min_height:
                mht_roots.append(root)
            else:
                min_height = root_height
                mht_roots = [root]
            # print(min_height, mht_roots)

        return mht_roots

    def dfs_height(self, node: int, root_to_leaf: List[int], nodes_to_visit: Set[int], min_height: int) -> int:
        root_to_leaf.append(node)
        nodes_to_visit.remove(node)

        if len(root_to_leaf) > min_height:
            return -1

        max_child_height = 0
        for child in self.edges[node]:
            if child in nodes_to_visit:
                child_height = self.dfs_height(
                    child, root_to_leaf, nodes_to_visit, min_height)
                if child_height == -1:
                    return -1
                max_child_height = max(max_child_height, child_height)

        root_to_leaf.pop()
        return max_child_height + 1

    def findMinHeightTrees(self, n: int, edges: List[List[int]]) -> List[int]:
        """
        (above DFS method exceeds time limit, another algo from solution)

        Time: O(N) - since it's a tree, it has (N-1) edges: O(V + E) -> O(N)
        Space: O(N)
        """
        # min_height
        # -> means we need to find the roots that are the closest to all other nodes
        # -> also means, at each layer from bottom up, we want to have as many nodes as possible at bottom,
        #    at each layer bottom up, we find all possible (relative) leaf nodes
        self.formGraph(n, edges)
        # time O(V + E), space O(V + E)
        double_edges = {node: set(self.edges[node]) for node in range(n)}

        bottom_nodes = set([node for node in double_edges.keys()  # time O(V)
                            if len(double_edges[node]) <= 1])
        up_nodes = set()

        while len(double_edges) > 2:
            for node in bottom_nodes:  # time O(V) in total
                child = list(double_edges[node])[0]
                double_edges[child].remove(node)
                if len(double_edges[child]) == 1:
                    up_nodes.add(child)
                double_edges.pop(node)
            bottom_nodes = up_nodes.copy()
            up_nodes = set()

        return [node for node in double_edges.keys()]
