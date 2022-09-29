from typing import Dict, List, Tuple


class Solution:
    def makeChangeRec(self, change: int):
        """
        (RECURSIVE)
        Given an integer representing a given amount of change, write a function to compute the
        total number of coins required to make that amount of change.
        You can assume that there is always a 1¢ coin. (assuming American coins: 1, 5, 10, and 25 cents)

        Time: O(C^4) according to the book but I don't see exactly why. (why not 4^C?)
        Space: O(C)
        """
        coins = set({1, 5, 10, 25})
        if change in coins:
            return 1

        min_count = None
        for coin in list(coins):
            if change - coin > 0:
                count = 1 + self.makeChangeRec(change - coin)
                if not min_count or count < min_count:
                    min_count = count
        return min_count

    def makeChangeDP(self, change: int):
        """
        (Dynamic Programming)

        Time: O(C * 4) -> O(C)
        Space: O(C)
        """
        coins = set({1, 5, 10, 25})
        if change in coins:
            return 1

        change_count = {c: 1 for c in coins}
        while True:
            for cur_change in list(change_count.keys()):
                count = change_count[cur_change]
                for coin in coins:
                    new_change = cur_change + coin
                    if new_change not in change_count:
                        change_count[new_change] = count + 1
                    if new_change == change:
                        return change_count[new_change]

    def squareSubmatrixRec(self, matrix: List[List[int]]) -> int:
        """
        (RECURSIVE)
        Given a 2D boolean array, find the largest square subarray of true values.
        The return value should be the side length of the largest square subarray subarray.

        Time: O(3^(m+n) * m * n)
        Space: O(m+n)
        """
        max_len = 0
        for row in range(len(matrix)):
            for col in range(len(matrix[0])):
                max_len = max(
                    max_len, self.squareSubmatrixAtRec(matrix, row, col))

        return max_len

    def squareSubmatrixAtRec(self, matrix: List[List[int]], row: int, col: int) -> int:
        """
        (RECURSIVE)
        Helper function to find the biggest submatrix of all True starting from matrix[row][col].

        Time: O(3^(m+n))
        Space: O(m+n)
        """
        if row == len(matrix) or col == len(matrix[0]) or not matrix[row][col]:
            return 0
        return 1 + min(
            self.squareSubmatrixAtRec(matrix, row+1, col),
            self.squareSubmatrixAtRec(matrix, row, col+1),
            self.squareSubmatrixAtRec(matrix, row+1, col+1))

    def squareSubmatrixDP(self, matrix: List[List[int]]) -> int:
        """
        (Dynamic Programming)

        Time: O(m * n)
        Space: O(m * n)
        """
        max_len = 0

        num_rows, num_cols = len(matrix), len(matrix[0])
        max_len_at = [[0] * num_cols for _ in range(num_rows)]

        # the last row / col cells will always have submatrix of all Trues of 1 or 0
        for i in range(num_rows):
            max_len_at[i][num_cols-1] = 1 if matrix[i][num_cols-1] else 0
            max_len = max(max_len, max_len_at[i][num_cols-1])
        for j in range(num_cols-1):
            max_len_at[num_rows-1][j] = 1 if matrix[num_rows-1][j] else 0
            max_len = max(max_len, max_len_at[num_rows-1][j])

        # keep updating cells from bottom-right -> up-left
        row, col = num_rows-2, num_cols-2
        while row >= 0 and col >= 0:
            # update the col
            for i in range(row, -1, -1):
                if not matrix[i][col]:
                    max_len_at[i][col] = 0
                else:
                    max_len_at[i][col] = 1 + min(max_len_at[i+1][col],
                                                 max_len_at[i][col+1],
                                                 max_len_at[i+1][col+1])
                    max_len = max(max_len, max_len_at[i][col])

            # update the row
            for j in range(col-1, -1, -1):
                if not matrix[row][j]:
                    max_len_at[row][j] = 0
                else:
                    max_len_at[row][j] = 1 + min(max_len_at[row+1][j],
                                                 max_len_at[row][j+1],
                                                 max_len_at[row+1][j+1])
                    max_len = max(max_len, max_len_at[row][j])

            row -= 1
            col -= 1

        return max_len

    def knapsackRec(self, max_weight: int, items: List[Tuple[int, int]]) -> int:
        """
        (RECURSIVE)
        A knapsack carries max amount of weight. You have a set of items with their own weight and 
        a monetary value. Find the max amount of money it can carry.

        Time: O(2^N)
        Space: O(N)
        """
        if not items:
            return 0
        weight, value = items[0]
        res_max_value_not_include = self.knapsackRec(max_weight, items[1:])

        if weight > max_weight:
            return res_max_value_not_include

        res_max_value_include = self.knapsackRec(max_weight-weight, items[1:])
        return max(res_max_value_not_include, value + res_max_value_include)

    def knapsackDP(self, max_weight: int, items: List[Tuple[int, int]]) -> int:
        """
        (Dynamic Programming)

        Time: O(n * w)
        Space: O(n * w)
        """
        # max_values[i][w] == knapsack(w+1, items[i:])
        max_values = [[0] * max_weight for _ in items]

        # populate value from bottom-right to up-left
        for i in range(len(max_values)-1, -1, -1):
            for w in range(len(max_values[0])-1, -1, -1):
                cur_item_weight, cur_item_value = items[i]

                if i+1 == len(max_values):  # last row
                    exclude_cur = 0
                    include_cur = cur_item_value
                else:
                    exclude_cur = max_values[i+1][w]
                    if w - cur_item_weight >= 0:
                        include_cur = cur_item_value + \
                            max_values[i+1][w-cur_item_weight]
                    else:
                        include_cur = cur_item_value

                if cur_item_weight <= w+1:
                    max_values[i][w] = max(include_cur, exclude_cur)
                else:
                    max_values[i][w] = exclude_cur

        return max_values[0][max_weight-1]

    def knapsackMemo(self, max_weight: int, items: List[Tuple[int, int]]) -> int:
        """
        (RECURSIVE with memo)

        Time: < O(n * w)
        Space: < O(n * w)
        """
        max_values = {}
        self.knapsackRecMemo(max_weight, 0, items, max_values)
        return max_values[(max_weight, 0)]

    def knapsackRecMemo(self, max_weight: int, start: int, items: List[Tuple[int, int]], max_values: Dict[Tuple[int, int], int]):
        cur_weight, cur_value = items[start]

        if start == len(items)-1:
            if max_weight >= cur_weight:
                max_values[(max_weight, start)] = cur_value
            else:
                max_values[(max_weight, start)] = 0
            return

        if (max_weight, start+1) not in max_values:
            self.knapsackRecMemo(max_weight, start+1, items, max_values)
        exclude_cur = max_values[(max_weight, start+1)]

        if max_weight < cur_weight:
            max_values[(max_weight, start)] = exclude_cur
            return

        if (max_weight-cur_weight, start+1) not in max_values:
            self.knapsackRecMemo(max_weight-cur_weight,
                                 start+1, items, max_values)
        include_cur = cur_value + max_values[(max_weight-cur_weight, start+1)]
        max_values[(max_weight, start)] = max(include_cur, exclude_cur)

    def targetSumRec(self, nums: List[int], target: int) -> int:
        """
        Find the number of ways that you can add and subtract the values in nums to add up to target.
        e.x. nums = [1, 1, 1, 1, 1], target = 3 -> 5
             (1+1+1+1-1, 1+1+1-1+1, 1+1-1+1+1, 1-1+1+1+1, -1+1+1+1+1)

        Time: O(2^N)
        Space: O(N)
        """
        return self.targetSumRecHelper(nums, target, 0)

    def targetSumRecHelper(self, nums: List[int], target: int, cur_sum: int) -> int:
        # reached the end, check if cur_sum == target
        if not nums:
            return 1 if target == cur_sum else 0

        # first num +/-
        first_pos = self.targetSumRecHelper(
            nums[1:], target, cur_sum + nums[0])
        first_neg = self.targetSumRecHelper(
            nums[1:], target, cur_sum - nums[0])
        return first_pos + first_neg

    def targetSumDP(self, nums: List[int], target: int) -> int:
        """
        (Dynamic Programming)

        Time: O(sum(nums) * n)
        Space: O(sum(nums) * n)
        """
        sum_nums = sum(nums)

        # start_sum[i][j] = number of ways to sum to target from nums[i:], with current sum == j - sum(nums)
        # also means number of ways to sum to target - cur_sum from nums[i:]
        start_sum = [[0] * (2 * abs(sum_nums) + 1)
                     for _ in range(len(nums) + 1)]

        # populate from bottom to up
        for i in range(len(start_sum)-1, -1, -1):
            for j in range(len(start_sum[0])):
                if i == len(start_sum)-1:
                    start_sum[i][j] = 1 if j == target + sum_nums else 0
                else:
                    neg_me = start_sum[i+1][j-nums[i]] if j >= nums[i] else 0
                    pos_me = start_sum[i+1][j+nums[i]
                                            ] if j < len(start_sum[0]) - nums[i] else 0
                    start_sum[i][j] = neg_me + pos_me

        return start_sum[0][sum_nums]

    def targetSumMemo(self, nums: List[int], target: int) -> int:
        """
        (RECURSIVE with memo)

        Time: < O(sum(nums) * n)
        Space: < O(sum(nums) * n)
        """
        start_sum = {}  # (start, cur_sum) -> number of ways
        self.targetSumRecMemo(nums, target, 0, 0, start_sum)
        return start_sum[(0, 0)]

    def targetSumRecMemo(self, nums: List[int], target: int, start: int, cur_sum: int, start_sum: Dict[Tuple[int, int], int]):
        if start == len(nums):
            start_sum[(start, cur_sum)] = 1 if cur_sum == target else 0
            return

        # negative me
        if (start+1, cur_sum - nums[start]) not in start_sum:
            self.targetSumRecMemo(nums, target, start+1,
                                  cur_sum - nums[start], start_sum)
        neg_me = start_sum[(start+1, cur_sum - nums[start])]

        # positive me
        if (start+1, cur_sum + nums[start]) not in start_sum:
            self.targetSumRecMemo(nums, target, start+1,
                                  cur_sum + nums[start], start_sum)
        pos_me = start_sum[(start+1, cur_sum + nums[start])]

        start_sum[(start, cur_sum)] = neg_me + pos_me


if __name__ == "__main__":
    s = Solution()
    print(s.makeChangeRec(1))
    print(s.makeChangeRec(6))
    print(s.makeChangeRec(49))

    print(s.makeChangeDP(1))
    print(s.makeChangeDP(6))
    print(s.makeChangeDP(49))

    matrix1 = [
        [False, True, False, False],
        [True, True, True, True],
        [False, True, True, False]
    ]
    matrix2 = [
        [True, True, True, True, True],
        [True, True, True, True, False],
        [True, True, True, True, False],
        [True, True, True, True, False],
        [True, False, False, False, False]
    ]
    matrix3 = [
        [True, True, True, True, True],
        [True, True, True, True, False],
        [True, True, True, True, False],
        [False, True, True, True, False],
        [True, False, False, False, False]
    ]

    print(s.squareSubmatrixRec(matrix1))
    print(s.squareSubmatrixRec(matrix2))
    print(s.squareSubmatrixRec(matrix3))

    print(s.squareSubmatrixDP(matrix1))
    print(s.squareSubmatrixDP(matrix2))
    print(s.squareSubmatrixDP(matrix3))

    items1 = [(2, 6), (2, 10), (3, 12)]
    max_weight1 = 5
    items2 = [(4, 1), (5, 2), (1, 3)]
    max_weight2 = 4
    items3 = [(4, 1), (5, 2), (6, 3)]
    max_weight3 = 3

    print(s.knapsackRec(max_weight1, items1))
    print(s.knapsackRec(max_weight2, items2))
    print(s.knapsackRec(max_weight3, items3))

    print(s.knapsackDP(max_weight1, items1))
    print(s.knapsackDP(max_weight2, items2))
    print(s.knapsackDP(max_weight3, items3))

    print(s.knapsackMemo(max_weight1, items1))
    print(s.knapsackMemo(max_weight2, items2))
    print(s.knapsackMemo(max_weight3, items3))

    nums1 = [1, 1, 1, 1, 1]
    target1 = 3
    nums2 = [1, 2, 3, 4, 5, 6, 7, 8]
    target2 = 20
    nums3 = [1, 1, 1, 1, 1]
    target3 = 9

    print(s.targetSumRec(nums1, target1))
    print(s.targetSumRec(nums2, target2))
    print(s.targetSumRec(nums3, target3))

    print(s.targetSumDP(nums1, target1))
    print(s.targetSumDP(nums2, target2))
    print(s.targetSumDP(nums3, target3))

    print(s.targetSumMemo(nums1, target1))
    print(s.targetSumMemo(nums2, target2))
    print(s.targetSumMemo(nums3, target3))
