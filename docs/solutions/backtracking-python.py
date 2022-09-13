import collections
from enum import unique
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

    def isPalindrome(self, substr: str) -> bool:
        """
        O(N)
        """
        i = 0
        j = len(substr) - 1
        while i < j:
            if substr[i] != substr[j]:
                return False
            i += 1
            j -= 1
        return True

    def partition(self, s: str) -> List[List[str]]:
        """
        (RECURSIVE)
        Partition s such that every substring is a palindrome.
        e.x. s = "aab" -> [["a","a","b"],["aa","b"]]

        Time: O(N * 2^N) - worst case all subsets are palindromes
        Space: O(N * 2^N)
        """
        all_pals = []
        if self.isPalindrome(s):
            all_pals.append([s])

        for i in range(len(s)-1, 0, -1):
            last_substr = s[i:]
            if self.isPalindrome(last_substr):
                all_pals += [p + [last_substr] for p in self.partition(s[:i])]

        return all_pals

    def partition(self, s: str) -> List[List[str]]:
        """
        (BACKTRACKING)

        Time: O(N * 2^N) - worst case all subsets are palindromes
        Space: O(N) - prefix at max N items, O(N * 2^N) if counting all_pals passed around
        """
        all_pals = []
        self.partitionBT(s, [], 0, all_pals)
        return all_pals

    def partitionBT(self, s: str, prefix: List[str], start: int, allPals: List[List[str]]):
        if start > len(s) - 1:
            allPals.append(prefix)
            return

        for i in range(start, len(s)):
            if self.isPalindrome(s[start: i+1]):
                self.partitionBT(s, prefix + [s[start: i+1]], i+1, allPals)

    def permute(self, nums: List[int]) -> List[List[int]]:
        """
        (RECURSIVE)
        An array nums of distinct integers, return all the possible permutations.
        e.x. nums = [1,2,3] -> [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]

        Time: O(N!+(N-1)!+...+1)
        Space: O(N!)
        """
        if len(nums) == 1:
            return [nums]

        all_perms = []

        res_perms = self.permute(nums[1:])
        for perm in res_perms:
            for i in range(len(perm)):
                perm.insert(i, nums[0])
                all_perms.append(perm.copy())
                perm.pop(i)
            perm.append(nums[0])
            all_perms.append(perm.copy())

        return all_perms

    def permute(self, nums: List[int]) -> List[List[int]]:
        """
        (BACKTRACKING)

        Time: O(N!+(N-1)!+...+1)
        Space: O(N!) - O(N) without output
        """
        if len(nums) == 1:
            return [nums]

        all_nums = set(nums)
        all_perms = []
        self.permuteBT(all_nums, [], all_perms)
        return all_perms

    def permuteBT(self, all_nums: Set[int], prefix: List[int], all_perms: List[List[int]]):
        if len(prefix) == len(all_nums):
            all_perms.append(prefix)
            return

        res_nums = all_nums - set(prefix)
        for num in res_nums:
            self.permuteBT(all_nums, prefix + [num], all_perms)

    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        """
        (BACKTRACKING)
        Array of nums, that might contain duplicates, return all possible unique permutations in any order.
        e.x. nums = [1,1,2] -> [[1,1,2],[1,2,1],[2,1,1]]

        Time: O(N!+(N-1)!+...+1)
        Space: O(N!) - O(N) if not considering args passing around
        """
        if len(nums) == 1:
            return [nums]

        nums_count = collections.Counter(nums)

        all_perms = []
        self.permuteUniqueBT(len(nums), nums_count, [], all_perms)
        return all_perms

    def permuteUniqueBT(self, total_nums: int, nums_count: Dict[int, int], prefix: List[int], all_perms: List[List[int]]):
        if len(prefix) == total_nums:
            all_perms.append(prefix)
            return

        remain_nums = nums_count.copy()
        for n in prefix:
            remain_nums[n] -= 1
            if remain_nums[n] == 0:
                remain_nums.pop(n)

        for n in remain_nums.keys():
            self.permuteUniqueBT(total_nums, nums_count,
                                 prefix + [n], all_perms)

    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        """
        (RECURSIVE)

        Time: O(N!+(N-1)!+...+1)
        Space: O(N!)
        """
        if len(nums) == 1:
            return [nums]

        all_perms = []
        unique_nums = set(nums)
        for num in unique_nums:
            remain_nums = nums.copy()
            remain_nums.remove(num)
            remain_perms = self.permuteUnique(remain_nums)
            for perm in remain_perms:
                all_perms.append([num] + perm)

        return all_perms

    def generateParenthesis(self, n: int) -> List[str]:
        """
        (RECURSIVE)
        Given n pairs of parentheses, generate all combinations of well-formed parentheses.
        e.x. n = 3 -> ["((()))","(()())","(())()","()(())","()()()"]

        Time: O() - I don't know actually but in solution it says O(4^N / N^0.5)
        Space: O() - same and in solution it says O(4^N / N^0.5)
        """
        return self.generateParsRec(["(", ")"] * n, n, n)

    def generateParsRec(self, pars: List[str], n_left: int, n_right: int) -> List[str]:
        """
        Helper function to generate all combinations of semi well-formed parentheses.
        e.x. ["(", ")", ")"] -> ["())", ")()"]
        """
        if len(pars) == 1:
            return pars

        all_pars = []

        if n_left == 0:   # if all remaining pars are ")"
            heads = [")"]
        # we cannot let remaining part have more "(" than ")"
        elif n_left == n_right:
            heads = ["("]
        else:
            heads = ["(", ")"]

        for h in heads:
            remain_pars = pars.copy()
            remain_pars.remove(h)
            if h == ")":
                all_remain_pars = self.generateParsRec(
                    remain_pars, n_left, n_right-1)
            else:
                all_remain_pars = self.generateParsRec(
                    remain_pars, n_left-1, n_right)
            for par in all_remain_pars:
                all_pars.append(h + par)
        return all_pars

    def generateParenthesis(self, n: int) -> List[str]:
        """
        (BACKTRACKING)

        Time: O() - same and in solution it says O(4^N / N^0.5)
        Space: O() - same and in solution it says O(4^N / N^0.5)
        """
        all_pars = []
        par_count = {"(": n, ")": n}
        self.generateParsBT(par_count, "", n, all_pars)
        return all_pars

    def generateParsBT(self, par_count: Dict[str, int], prefix: str, n: int, all_pars: List[str]):
        if len(prefix) == 2 * n:
            all_pars.append(prefix)
            return

        par_count_remain = par_count.copy()
        for p in prefix:
            par_count_remain[p] -= 1

        if par_count_remain["("] == 0:
            heads = [")"]
        elif par_count_remain["("] == par_count_remain[")"]:
            heads = ["("]
        else:
            heads = ["(", ")"]

        for h in heads:
            self.generateParsBT(par_count, prefix + h, n, all_pars)

    def generateParenthesis(self, n: int) -> List[str]:
        """
        (Another RECURSIVE algo from solution)

        Time: O() - same and in solution it says O(4^N / N^0.5)
        Space: O() - same and in solution it says O(4^N / N^0.5)
        """
        # there must be a "(" at index 0, and a ")" at each 2*i + 1 as a closure ()
        # ()..., (.)..., (..)..., (...)...
        # for each closure of some "(" and ")", (seq_A)seq_B's seq_A and seq_B must be valid as well
        if n == 0:
            return [""]
        if n == 1:
            return ["()"]

        all_pars = []
        for i in range(n):
            seq_a = self.generateParenthesis(i)
            seq_b = self.generateParenthesis(n - (i + 1))
            for left in seq_a:
                for right in seq_b:
                    all_pars.append(f"({left}){right}")

        return all_pars
