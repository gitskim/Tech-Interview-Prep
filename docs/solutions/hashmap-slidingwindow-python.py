from calendar import firstweekday
import collections
from copy import copy
from email import charset
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
