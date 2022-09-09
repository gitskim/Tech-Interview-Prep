import collections
from sys import prefix
from typing import Dict, List, Set


class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        """
        A str digits from [2:9], return all possible letter combinations that the number could represent.
        Number -> letter mapping as phone keyboard.
        e.x. "23" -> ["ad","ae","af","bd","be","bf","cd","ce","cf"]

        Time: O(3^N)
        Space: O(3^N)
        """
        if not digits:
            return []

        # form list of char lists
        char_groups = []
        for digit in digits:  # O(N)
            num = int(digit)

            base_chars = "abc"
            if num == 7 or num == 9:
                base_chars = "abcd"

            offset = 0
            if num > 7:
                offset = 1

            char_groups.append([chr((num-2) * 3 + ord(c) + offset)
                               for c in base_chars])

        all_combs = []
        self.allCombinations(char_groups, "", 0, all_combs)
        return all_combs

    def allCombinations(self, charGroups: List[List[chr]], prefix: str, curIdx: int, allCombs: List[str]):
        """
        (BACKTRACKING)
        """
        if curIdx == len(charGroups) - 1:
            for c in charGroups[curIdx]:
                allCombs.append(prefix + c)
            return
        for c in charGroups[curIdx]:
            self.allCombinations(charGroups, prefix + c, curIdx + 1, allCombs)

    def allCombinations(self, charGroups: List[List[chr]]) -> List[str]:
        """
        (RECURSIVE)
        """
        if not charGroups:
            return []
        if len(charGroups) == 1:
            return charGroups[0]

        all_sub_strs = []
        for sub_str in self.allCombinations(charGroups[:-1]):
            for c in charGroups[-1]:
                all_sub_strs.append(sub_str + c)

        return all_sub_strs

    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        """
        (BACKTRACK)
        An int array nums that may contain duplicates, return all possible subsets (the power set).
        The solution set must not contain duplicate subsets.
        e.x. [1, 2, 2] -> [[],[1],[1,2],[1,2,2],[2],[2,2]]

        Time: O(2^N)
        Space: O(2^N * N)
        """
        # sort nums for easier dup check
        nums.sort()

        all_subsets = [[]]
        self.subsetsBacktrack(nums, [], 0, all_subsets)

        return all_subsets

    def subsetsBacktrack(self, nums: List[int], prefix: List[int], startIdx: int, all_subsets: Set[str]):
        if startIdx == len(nums):
            return

        cur_num = None
        for i in range(startIdx, len(nums)):
            # can think of as not adding dup number to it's children
            if nums[i] == cur_num:
                continue
            cur_num = nums[i]
            all_subsets.append(prefix + [nums[i]])
            self.subsetsBacktrack(nums, prefix + [nums[i]], i + 1, all_subsets)

    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        """
        (RECURSIVE)

        Time: O(2^N)
        Space: O(2^N * N)
        """
        nums.sort()
        return self.subsetsRecursive(nums)

    def subsetsRecursive(self, nums: List[int]) -> List[List[int]]:
        if not nums:
            return [[]]

        # find the last index where nums[i+1:] are all dup numbers - O(N) in total
        first_dup_i = 0
        for i in range(len(nums)-1, -1, -1):
            if nums[i] != nums[-1]:
                first_dup_i = i + 1
                break

        # get all subsets for previous non dup numbers
        all_subsets = self.subsetsRecursive(nums[:first_dup_i])

        # form new subsets of prev subsets + new num once
        prev_subsets_len = len(all_subsets)
        for i in range(prev_subsets_len):  # O(2^N)
            all_subsets.append(all_subsets[i] + [nums[-1]])

        # then for each dup number, form subsets of the latest prev_subsets_len subsets + new num
        for _ in range(first_dup_i+1, len(nums)):
            cur_subsets_len = len(all_subsets)
            for i in range(prev_subsets_len):
                all_subsets.append(
                    all_subsets[cur_subsets_len-prev_subsets_len+i] + [nums[-1]])

        return all_subsets

    def combine(self, n: int, k: int) -> List[List[int]]:
        """
        (RECURSIVE)
        int n and k -> all possible combinations of k numbers chosen from range [1, n]
        (1 <= n <= 20, 1 <= k <= n)
        e.x. n = 4, k = 2 -> 2 nums from [1, 2, 3, 4] -> [[1,2],[1,3],[1,4],[2,3],[2,4],[3,4]]

        Time: O(n * k) - (actually I am not really sure about this)
        Space: O(n!/(k!(n-k)!)) - (combination of (n, k))
        """
        # e.x. take 1 from [1,2,3] -> [[1], [2], [3]]
        if k == 1:
            return [[x] for x in range(1, n+1)]
        # e.x. take 3 from [1,2,3] -> [[1, 2, 3]]
        if k == n:
            return [[x for x in range(1, n+1)]]

        # e.x. take 2 from [1, 2, 3, 4] ->
        #      take 1 from [1, 2, 3] combined with [4] + take 2 from [1, 2, 3]
        return [prev + [n] for prev in self.combine(n-1, k-1)] + self.combine(n-1, k)

    def combine(self, n: int, k: int) -> List[List[int]]:
        """
        (BACKTRACKING)

        Time: O(k * (n-k)^2) (?)
        Space: O(n!/(k!(n-k)!))
        """
        all_combs = []
        self.subsetsK(n, k, [], 1, all_combs)
        return all_combs

    def subsetsK(self, n: int, k: int, prefix: List[int], startN: int, allCombs: List[List[int]]):
        if len(prefix) == k:
            allCombs.append(prefix)
            return

        for i in range(startN, n+1):
            self.subsetsK(n, k, prefix + [i], i+1, allCombs)

    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        """
        (RECURSIVE)
        Array of distinct int candidates and an int target 
          -> all unique combinations of candidates that sum to target
        The same number may be chosen from candidates an unlimited number of times.
        e.x. candidates = [2,3,6,7], target = 7 -> [[2,2,3],[7]]

        Time: O() - I actually don't know
        Space: O() - same
        """
        if not candidates:
            return []

        last_num = candidates[-1]
        q, r = divmod(target, last_num)

        if len(candidates) == 1:
            if r == 0:
                return [[last_num] * q]
            return []

        all_sums = []
        for i in range(q + 1):
            prev_sums = self.combinationSum(
                candidates[:-1], target - last_num * i)
            if prev_sums:
                all_sums += [ps + [last_num] * i for ps in prev_sums]
        return all_sums

    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        """
        (ANOTHER RECURSIVE)

        Time: O() - ?
        Space: O() - same
        """
        candidates.sort()
        return self.combinationRec(candidates, target)

    def combinationRec(self, candidates: List[int], target: int) -> List[List[int]]:
        if not candidates:
            return []

        all_sums = []
        first_num = candidates[0]
        q, _ = divmod(target, first_num)

        for i in range(q + 1):
            prefix_sum = first_num * i
            if prefix_sum > target:
                return all_sums
            if prefix_sum == target:
                all_sums += [[first_num] * i]
                return all_sums

            follow_sums = self.combinationSum(
                candidates[1:], target - prefix_sum)
            if follow_sums:
                all_sums += [[first_num] * i + fs for fs in follow_sums]
        return all_sums

    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        """
        (BACKTRACKING)

        Time: O() - ?
        Space: O() - same
        """
        if not candidates:
            return []

        candidates.sort()
        all_sums = []
        self.combinationBT(candidates, target, [], 0, all_sums)
        return all_sums

    def combinationBT(self, candidates: List[int], target: int, prefix: List[int], start: int, allSums: List[List[int]]):
        cur_sum = sum(prefix)
        if cur_sum == target:
            allSums.append(prefix)
            return

        for i in range(start, len(candidates)):
            # already sorted, stop when impossible to form anything == target
            if cur_sum + candidates[i] > target:
                return
            self.combinationBT(candidates, target, prefix +
                               [candidates[i]], i, allSums)
