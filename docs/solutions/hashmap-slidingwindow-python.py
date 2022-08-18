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
