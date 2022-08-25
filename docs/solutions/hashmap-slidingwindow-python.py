from calendar import firstweekday
import collections
from copy import copy
from ctypes.wintypes import SMALL_RECT
from email import charset
from enum import unique
from multiprocessing import allow_connection_pickling
from pdb import pm
from pickle import TRUE
from textwrap import wrap
from typing import List


class Solution:

    def twoSum(self, nums: List[int], target: int) -> List[int]:
        """
        TwoSum
        Time: O(N)
        Space: O(N)
        """
        scanned = {}
        for i in range(len(nums)):
            comp = target - nums[i]
            if comp in scanned:
                return [scanned[comp], i]
            scanned[nums[i]] = i

    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        """
        TwoSum of non-descending ordered list
        Time: O(N)
        Space: O(1)
        """
        i = 0
        j = len(numbers) - 1
        while i < j:
            curSum = numbers[i] + numbers[j]
            if curSum == target:
                return [i+1, j+1]
            if curSum > target:
                j -= 1
            else:
                i += 1

    def threeSum(self, nums: List[int]) -> List[List[int]]:
        """
        ThreeSum
        Time: O(N^2)
        Space: O(N)
        """
        nums.sort()    # NlogN
        triplets = []
        for i in range(len(nums)):    # N
            if i > 0 and nums[i] == nums[i-1]:
                continue
            twoSumCouples = self.twoSumAll(nums[i+1:], 0-nums[i])  # N
            for couple in twoSumCouples:  # N
                triplets.append([nums[i]] + couple)
        return triplets

    def twoSumAll(self, nums: List[int], targetSum: int) -> List[List[int]]:
        """
        TwoSum of non-descending ordered list - all possible couples
        Time: O(N)
        Space: O(N)
        """
        i = 0
        j = len(nums) - 1
        selNum = set()
        while i < j:
            curSum = nums[i] + nums[j]
            if curSum == targetSum:
                if nums[i] not in selNum:
                    selNum.add(nums[i])
                i += 1
                j -= 1
            elif curSum < targetSum:
                i += 1
            else:
                j -= 1
        return [[n, targetSum - n] for n in selNum]

    def minWindow(self, s: str, t: str) -> str:
        """
        2 strings s and t of lengths m and n -> minimum substring of s that contains every character in t
        Time: O(m+n) (techinically O(m) in my understanding)
        Space: O(m)
        """
        minW = ""

        if len(t) > len(s):
            return minW

        # collect number of occurances of all characters in t
        tMap = collections.Counter(t)   # O(n) < O(m)

        tInS = []  # list of char's index in S for char in t
        allChars = set(tMap.keys())  # all chars to include in minW

        i = 0
        for i in range(len(s)):   # O(m)
            c = s[i]
            if c in tMap:
                tMap[c] -= 1
                tInS.append(i)
                # collected all occurance of a certain char
                if c in allChars and tMap[c] <= 0:
                    allChars.remove(c)
                # collected all occurance of all chars
                if not allChars:
                    # remove unnecessary chars at front
                    while True:   # < O(m) in all loops together
                        firstChar = s[tInS[0]]
                        if tMap[firstChar] >= 0:
                            break
                        tMap[firstChar] += 1
                        tInS.pop(0)
                    # form a window that contains all chars in t and check length
                    if minW == "" or tInS[-1] - tInS[0] + 1 < len(minW):
                        minW = s[tInS[0]: tInS[-1]+1]
                        # print(minW)
                        if len(minW) == len(t):
                            return minW
                    # pop one char at front and find next window that contains it
                    poppedC = s[tInS.pop(0)]
                    tMap[poppedC] += 1
                    allChars.add(poppedC)
                    # print(tMap)

        return minW

    def findAnagrams(self, s: str, p: str) -> List[int]:
        """
        2 strings s and p (len m and n) -> array of all the start indices of p's anagrams in s
        Time: O(m+n)
        Space: O(m)
        """
        if not s or len(p) > len(s):
            return []

        allStarts = []  # output list of indices of p's anagrams
        pMap = collections.Counter(p)   # O(n) - num of occurances of p's chars
        allChars = set(pMap.keys())  # all chars in p

        pInS = []  # index of char in s for all chars in p
        for i in range(len(s)):    # O(m)
            c = s[i]
            if c in pMap:
                pMap[c] -= 1
                pInS.append(i)
                # collected all occurances of char c
                if c in allChars and pMap[c] <= 0:
                    allChars.remove(c)
                # collected all occurances of all chars
                if not allChars:
                    # remove unnecessary chars at front
                    while True:   # < O(m) for all the loops
                        firstChar = s[pInS[0]]
                        if pMap[firstChar] >= 0:
                            break
                        pInS.pop(0)
                        pMap[firstChar] += 1
                    # if substring doesn't have any other char
                    if pInS[-1] - pInS[0] + 1 == len(p):
                        allStarts.append(pInS[0])
                    # pop start and find next substring that has all chars in p
                    popped = s[pInS.pop(0)]
                    pMap[popped] += 1
                    allChars.add(popped)

        return allStarts

    def findAnagrams(self, s: str, p: str) -> List[int]:
        """
        (Another solution from ideas in discussion)
        2 strings s and p (len m and n) -> array of all the start indices of p's anagrams in s
        Time: O(m+n)
        Space: O(1) - <=26 keys in all dicts (not including return list allStarts)
        """
        if not s or len(p) > len(s):
            return []

        allStarts = []  # output list of indices of p's anagrams
        pMap = collections.Counter(p)   # O(n) - num of occurances of p's chars
        winMap = {}    # num of occurances of chars in window

        # form initial window
        for i in range(len(p)):    # O(n)
            if s[i] not in winMap:
                winMap[s[i]] = 1
            else:
                winMap[s[i]] += 1
        if winMap == pMap:
            allStarts.append(0)

        # slide window
        for i in range(len(p), len(s)):    # O(m)
            # delete first char
            firstChar = s[i-len(p)]
            winMap[firstChar] -= 1
            if winMap[firstChar] == 0:
                winMap.pop(firstChar)
            # add new char
            if s[i] not in winMap:
                winMap[s[i]] = 1
            else:
                winMap[s[i]] += 1
            # compare
            if winMap == pMap:
                allStarts.append(i-len(p)+1)

        return allStarts

    def findSubstring(self, s: str, words: List[str]) -> List[int]:
        """
        A string s (len n) and an array of strings words (len a * b)
            -> array of all the start indices of concatenated substrings (all strs in words) in s
        Time: O(a*n) in worst case
        Space: O(a*b)
        """
        if len(s) < len(words) * len(words[0]):
            return []

        allStarts = []  # output list of indices concatenated substrings
        # O(a) - num of occurances of word in words
        wordMap = collections.Counter(words)
        wordLen = len(words[0])    # all words will be the same length

        # O(a*b) - get sum of all chars in words to potentially speed up the check
        wordCharSum = 0
        for word in words:
            for c in word:
                wordCharSum += ord(c)

        # O(a*b) - get sum of all chars in first len(words) * wordLen - 1 chars of s
        charSum = 0
        for i in range(len(words) * wordLen - 1):
            charSum += ord(s[i])

        # O(n) - form and slide window
        for i in range(len(words) * wordLen - 1, len(s)):
            # first check if sums of all chars are the same
            idxToDel = i - len(words) * wordLen
            if idxToDel >= 0:
                charSum -= ord(s[idxToDel])
            charSum += ord(s[i])
            if charSum != wordCharSum:
                continue

            # if first check passed, form a word list to compare
            winMap = {}    # num of occurances of words in window
            notSame = False
            for n in range(len(words)):    # O(a)
                word = s[i+1-(n+1)*wordLen:i+1-n*wordLen]
                if word not in wordMap:
                    notSame = True
                    break
                if word not in winMap:
                    winMap[word] = 1
                else:
                    winMap[word] += 1
            # compare
            if not notSame and winMap == wordMap:
                allStarts.append(i - (len(words) * wordLen - 1))

        return allStarts

    def findSubstring(self, s: str, words: List[str]) -> List[int]:
        """
        (Another sliding window solution from idea in discussion)
        A string s (len n) and an array of strings words (len a * b)
            -> array of all the start indices of concatenated substrings (all strs in words) in s
        Time: O(a*b+n)
        Space: O(a)

        ps. switching to underscore variable naming from this solution on
        """
        num_words = len(words)
        word_length = len(words[0])

        if len(s) < num_words * word_length:
            return []

        all_starts = []  # output list of indices concatenated substrings
        # O(a) - num of occurances of word in words
        word_count = collections.Counter(words)

        for i in range(word_length):  # O(b)
            word_to_collect = word_count.copy()  # O(a)
            words_remain = set(word_to_collect.keys())
            for j in range(i, len(s)-word_length+1, word_length):  # O(n/b)
                # check word starting from j
                word = s[j:j+word_length]
                if word in word_to_collect:
                    word_to_collect[word] -= 1
                    # collected this word
                    if word in words_remain and word_to_collect[word] <= 0:
                        words_remain.remove(word)
                # collected all words
                if not words_remain:
                    all_starts.append(j-(num_words-1)*word_length)
                # pop first one to create space for next check
                first_index = j - (num_words-1) * word_length
                if first_index >= 0:
                    first_word = s[first_index: first_index+word_length]
                    if first_word in word_to_collect:
                        word_to_collect[first_word] += 1
                        if word_to_collect[first_word] > 0:
                            words_remain.add(first_word)

        return all_starts

    def lengthOfLongestSubstringTwoDistinct(self, s: str) -> int:
        """
        Find the length of the longest substring that contains at most two distinct characters in s.
        Time: O(n)
        Space: O(1)
        """
        if len(s) < 3:
            return len(s)

        longest_substr_length = 0
        start = end = 0  # first and last index of current substr
        char_count = {}  # occurances of each char in current substr

        for end in range(len(s)):  # O(n)
            # add in a new char from right
            if s[end] not in char_count:
                char_count[s[end]] = 1
            else:
                char_count[s[end]] += 1

            # if we have >2 unique chars in substr, shrink from left
            while start < len(s) - 1 and len(char_count) > 2:  # O(n) in total
                char_count[s[start]] -= 1
                if char_count[s[start]] == 0:
                    char_count.pop(s[start])
                start += 1

            if len(char_count) < 3 and end - start + 1 > longest_substr_length:
                longest_substr_length = end - start + 1

        return longest_substr_length

    def lengthOfLongestSubstringTwoDistinct(self, s: str) -> int:
        """
        (Similar algorithm from solution but left side shrinks faster)
        Find the length of the longest substring that contains at most two distinct characters in s.
        Time: O(n)
        Space: O(1)
        """
        if len(s) < 3:
            return len(s)

        longest_substr_length = 0
        start = end = 0  # first and last index of current substr
        # rightmost index for each char in current substr, at most 3 keys
        char_rightmost_idx = collections.defaultdict()

        for end in range(len(s)):
            char_rightmost_idx[s[end]] = end

            # if we have >2 unique chars in substr, drop one unique char from left, will come to 2 again
            if len(char_rightmost_idx) > 2:
                del_idx = min(char_rightmost_idx.values())
                char_rightmost_idx.pop(s[del_idx])
                start = del_idx + 1

            longest_substr_length = max(end - start + 1, longest_substr_length)

        return longest_substr_length

    def lengthOfLongestSubstringKDistinct(self, s: str, k: int) -> int:
        """
        Find the length of the longest substring that contains at most k distinct characters in s.
        Time: O(n*k) - solution uses ordereddict to bring down to O(n)
        Space: O(k)
        """
        if not k:
            return 0
        if len(s) < k + 1:
            return len(s)

        longest_substr_length = 0
        start = 0
        char_rightmost_idx = collections.defaultdict()

        for end in range(len(s)):
            # update char's rightmost idx
            char_rightmost_idx[s[end]] = end
            # if have >k unique chars, remove from left until one char removed from substr
            if len(char_rightmost_idx) > k:
                idx_del = min(char_rightmost_idx.values())  # O(k)
                char_rightmost_idx.pop(s[idx_del])
                start = idx_del + 1
            longest_substr_length = max(longest_substr_length, end - start + 1)

        return longest_substr_length

    def lengthOfLongestSubstringKDistinct(self, s: str, k: int) -> int:
        """
        (Another algo similar to the original implementation for when k=2)
        Find the length of the longest substring that contains at most k distinct characters in s.
        Time: O(n)
        Space: O(k)
        """
        if not k:
            return 0
        if len(s) < k + 1:
            return len(s)

        longest_substr_length = 0
        start = 0
        char_count = {}  # at most k+1 keys

        for end in range(len(s)):  # O(n)
            if s[end] not in char_count:
                char_count[s[end]] = 1
            else:
                char_count[s[end]] += 1

            while len(char_count) > k:  # O(n) in total
                char_count[s[start]] -= 1
                if char_count[s[start]] == 0:
                    char_count.pop(s[start])
                start += 1

            longest_substr_length = max(longest_substr_length, end - start + 1)

        return longest_substr_length

    def subarraysWithKDistinct(self, nums: List[int], k: int) -> int:
        """
        (From an idea from discussion session)
        An int array nums and an int k, return the number of good subarrays of nums.
        A good array is an array where the number of different integers in that array is exactly k.
        e.x. nums = [1,2,1,3,4], k = 3 -> return 3 ([1,2,1,3], [2,1,3], [1,3,4])
        Time: O(n)
        Space: O(k)
        """
        if k == 0 or len(nums) < k:
            return 0

        # number of subarrays with at most k different integers
        #     - number of subarrays with at most k-1 different integers
        #     = number of subarrays with exactly k different integers
        return self.subarraysWithAtMostKDistinct(nums, k) - self.subarraysWithAtMostKDistinct(nums, k-1)

    def subarraysWithAtMostKDistinct(self, nums: List[int], k: int) -> int:
        """
        Helper function to count number of subarrays in array nums that contains <=k different integers.
        Time: O(n)
        Space: O(k)
        """
        if k == 0 or not nums:
            return 0

        valid_subs = 0  # num of valid subarrays
        start = 0  # start of current subarray
        num_count = {}  # occurances of each number in current subarray

        for end in range(len(nums)):  # O(n)
            if nums[end] not in num_count:
                num_count[nums[end]] = 1
            else:
                num_count[nums[end]] += 1

            # shrink from left till at most k different numbers in array
            while len(num_count) > k:  # O(n) in total
                num_count[nums[start]] -= 1
                if num_count[nums[start]] == 0:
                    num_count.pop(nums[start])
                start += 1

            # at this point, nums[start: end+1] has exactly k different integers
            # e.x. nums[start: end+1] is like [x_s, x_s+1, ..., x_e]
            #   -> then all subarrays of this array has <= k different integers
            #   -> [x_s, x_s+1, ..., x_e-1]'s subarrays has been counted when end <= e-1
            #   -> just need to add: (e - s + 1) new subarrays
            #      [x_s, x_s+1, ..., x_e], [x_s+1, ..., x_e], [x_s+2, ..., x_e], ..., [x_e-1, x_e], [x_e]
            valid_subs += end - start + 1

        return valid_subs

    def subarraysWithKDistinct(self, nums: List[int], k: int) -> int:
        """
        (Algorithm from solution)
        An int array nums and an int k, return the number of good subarrays of nums.
        A good array is an array where the number of different integers in that array is exactly k.
        e.x. nums = [1,2,1,3,4], k = 3 -> return 3 ([1,2,1,3], [2,1,3], [1,3,4])
        Time: O(n)
        Space: O(k)
        """
        if not k or len(nums) < k:
            return 0

        valid_subs = 0  # num of valid subarrays
        start_greater_k = start_ge_k = 0  # start index of subarrays
        num_count_greater = {}  # occurances of each number in current subarray that diff nums > k
        num_count_ge = {}  # occurances of each number in current subarray that diff nums >= k

        for end in range(len(nums)):  # O(n)
            # add in a new number from right for both num_counts
            if nums[end] not in num_count_greater:
                num_count_greater[nums[end]] = 1
            else:
                num_count_greater[nums[end]] += 1
            if nums[end] not in num_count_ge:
                num_count_ge[nums[end]] = 1
            else:
                num_count_ge[nums[end]] += 1

            # if we have >k unique numbers in subarray, shrink from left
            # by the end start_greater_k will be at the first index where nums[start: end+1] has k diff nums
            # all nums[start < start_greater_k : end+1] will have >k diff nums
            while len(num_count_greater) > k:
                num_count_greater[nums[start_greater_k]] -= 1
                if num_count_greater[nums[start_greater_k]] == 0:
                    num_count_greater.pop(nums[start_greater_k])
                start_greater_k += 1

            # if we have >=k unique numbers in subarray, shrink from left
            # by the end start_ge_k will be at the first index where nums[start: end+1] has <k diff nums
            # all nums[start < start_ge_k : end+1] will have >=k diff nums
            while len(num_count_ge) >= k:
                num_count_ge[nums[start_ge_k]] -= 1
                if num_count_ge[nums[start_ge_k]] == 0:
                    num_count_ge.pop(nums[start_ge_k])
                start_ge_k += 1

            # the only possible cases where diff nums == k for this end are:
            #   nums[start_greater_k: end+1], nums[start_greater_k+1: end+1], ..., nums[start_ge_k-1: end+1]
            valid_subs += start_ge_k - start_greater_k

        return valid_subs
