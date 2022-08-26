import random
from re import T
from typing import List


class Solution:

    def findKthLargest(self, nums: List[int], k: int) -> int:
        """
        Find the kth largest element in the array
        Time: O(N) - N + N/2 + N/4 + N/8 + ... -> 2N
        Space: O(1)
        """
        random.shuffle(nums)
        start = 0
        end = len(nums) - 1
        k_largest_index = len(nums) - k  # where the partition index should be

        while start < end + 1:
            # nums[:p_smaller] < partition_value, nums[p_greater:] > partition_value
            p_smaller, p_greater = self.partitionNumsThreeWays(
                nums, start, end)
            if p_smaller <= k_largest_index < p_greater:
                return nums[p_smaller]
            if k_largest_index < p_smaller:
                end = p_smaller - 1
            else:
                start = p_greater
        return nums[start]

    def partitionNumsThreeWays(self, nums: List[int], low: int, high: int) -> List[int]:
        """
        Helper function to partition the nums in place (3-way). Partition value is nums[low].

        Args:
            nums (List[int]): list to partition, will be changed in-place
            low (int): start index of the subarray to partition
            high (int): end index of the subarray to partition

        Returns:
            List[int]: the partition points indices where nums[low: p1] < nums[p1: p2] == partition_value < nums[p2: high+1]

        Time: O(N) -- (high-low)
        Space: O(1)
        """
        def swap(nums: List[int], a: int, b: int):
            temp_num = nums[a]
            nums[a] = nums[b]
            nums[b] = temp_num

        # sort and return directly if subarray length < 3
        if high - low < 2:
            if nums[high] < nums[low]:
                swap(nums, low, high)
                return [high, high+1]
            if nums[low] == nums[high]:
                return [low, high+1]
            return [low, high]

        # start partition 3 ways
        # nums[:p_smaller] < p_val, nums[:p_smallerequal] <= p_val
        p_smaller = p_smallerequal = low + 1
        p_greater = high  # nums[p_greater+1:] > p_val
        partition_val = nums[low]

        while p_smallerequal < p_greater + 1:
            if nums[p_smallerequal] < partition_val:
                swap(nums, p_smallerequal, p_smaller)
                # after swap, nums[p_smaller] < p_val, nums[p_smallerequal] <= p_val
                #   -> at first p_smallerequal == p_smaller, no swap technically, current num < p_val
                #   -> then after seeing some nums == p_val,
                #      before swap p_smaller would be on the first num == p_val,
                #      after swap nums[p_smallerequal] == p_val
                p_smaller += 1
                p_smallerequal += 1
            elif nums[p_smallerequal] == partition_val:
                p_smallerequal += 1
            else:
                swap(nums, p_smallerequal, p_greater)
                # after swap, nums[p_greater] > p_val, nums[p_smallequal] unknown
                p_greater -= 1

        # if p_val happens to be the greatest val in subarray
        # p_smaller == high + 1 or first num == p_val
        if partition_val >= nums[high]:
            swap(nums, low, p_smaller-1)
            return [p_smaller-1, high+1]
        # if p_val happens to be the smallest val in subarray
        # p_greater == low or last num == p_val
        if partition_val <= nums[low+1]:
            return [low, p_greater+1]

        swap(nums, low, p_smaller-1)
        return [p_smaller-1, p_greater+1]

    def findKthLargest(self, nums: List[int], k: int) -> int:
        """
        Find the kth largest element in the array
        Time: O(N) - N + N/2 + N/4 + N/8 + ... -> 2N
        Space: O(1)
        """
        random.shuffle(nums)
        start = 0
        end = len(nums) - 1
        k_largest_index = len(nums) - k  # where the partition index should be

        while start < end + 1:
            p_smaller = self.partitionNums(nums, start, end)
            if k_largest_index == p_smaller:
                return nums[p_smaller]
            if k_largest_index < p_smaller:
                end = p_smaller - 1
            else:
                start = p_smaller + 1

        return nums[start]

    def partitionNums(self, nums: List[int], low: int, high: int) -> int:
        """
        Helper function to partition the nums in place (2-way). Partition value is nums[low].

        Args:
            nums (List[int]): list to partition, will be changed in-place
            low (int): start index of the subarray to partition
            high (int): end index of the subarray to partition

        Returns:
            int: the partition point index where nums[low: p] < p_val <= nums[p: high+1]

        Time: O(N)
        Space: O(1)
        """
        def swap(nums: List[int], a: int, b: int):
            temp_num = nums[a]
            nums[a] = nums[b]
            nums[b] = temp_num

        # if subarray length is <3
        if high - low < 2:
            if nums[high] < nums[low]:
                swap(nums, low, high)
                return high
            return low

        partition_val = nums[low]
        start = low + 1  # nums[:start] should be <p_val
        end = high  # nums[end:] should be >=p_val
        while True:
            while start < end + 1 and nums[start] < partition_val:
                start += 1
            # at this point nums[start] >= p_val or start == end
            while start < end + 1 and nums[end] >= partition_val:
                end -= 1
            # at this point nums[end] < p_val or end == start
            if start > end:
                break

            swap(nums, start, end)

        # if nums[low] is the smallest within subarray
        if partition_val <= nums[low+1]:
            return low

        swap(nums, low, start - 1)
        return start - 1
