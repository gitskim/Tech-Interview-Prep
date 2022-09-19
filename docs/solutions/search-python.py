import math
from typing import List


class Solution:
    def findClosestElements(self, arr: List[int], k: int, x: int) -> List[int]:
        """
        A sorted int array arr, two ints k and x, return the k closest integers to x in the array. 
        The result should also be sorted in ascending order.

        Time: O(logN + k)
        Space: O(1)
        """
        start = end = self.binarySearch(arr, x)
        while end - start < k and start >= 0 and end < len(arr):
            if abs(arr[start] - x) <= abs(arr[end] - x):
                start -= 1
            else:
                end += 1

        if start < 0:
            return arr[:k]
        if end >= len(arr):
            return arr[-k:]
        if abs(arr[start] - x) <= abs(arr[end] - x):
            return arr[start: end]
        return arr[start+1: end+1]

    def binarySearch(self, arr: List[int], target: int) -> int:
        """
        (ITERATIVE)
        Helper function to find the closest number to target and return its index.

        Time: O(logN)
        Space: O(1)
        """
        start = 0
        end = len(arr) - 1

        while start <= end:
            cur = math.floor((start + end) / 2)
            if arr[cur] == target:
                return cur
            if arr[cur] > target:
                end = cur - 1
            else:
                start = cur + 1

        # no match, return the closest one
        if end < 0:
            return start
        if start >= len(arr):
            return end
        if abs(arr[start] - target) < abs(arr[end] - target):
            return start
        return end

    def binarySearch(self, arr: List[int], target: int) -> int:
        """
        (RECURSIVE)

        Time: O(logN)
        Space: O(logN)
        """
        if len(arr) == 1:
            return 0
        if len(arr) == 2:
            if abs(arr[0] - target) < abs(arr[1] - target):
                return 0
            return 1

        cur = math.floor((len(arr)-1) / 2)
        if arr[cur] == target:
            return cur
        if arr[cur] > target:
            return self.binarySearch(arr[:cur+1], target)
        return cur + 1 + self.binarySearch(arr[cur+1:], target)

    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        """
        (ITERATIVE)
        Search for a value target in an m x n integer matrix matrix. 
        This matrix has the following properties:
        - Integers in each row are sorted from left to right.
        - The first integer of each row is greater than the last integer of the previous row.

        Time: O(logM + logN)
        Space: O(1)
        """
        start_row = 0
        end_row = len(matrix) - 1

        while start_row <= end_row:
            cur_row = math.floor((start_row + end_row) / 2)
            if matrix[cur_row][0] > target:
                end_row = cur_row - 1
            elif matrix[cur_row][-1] < target:
                start_row = cur_row + 1
            else:
                start_col = 0
                end_col = len(matrix[0]) - 1
                while start_col <= end_col:
                    cur_col = math.floor((start_col + end_col) / 2)
                    if matrix[cur_row][cur_col] == target:
                        return True
                    if matrix[cur_row][cur_col] < target:
                        start_col = cur_col + 1
                    else:
                        end_col = cur_col - 1
                return False
        return False

    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        """
        (RECURSIVE)

        Time: O(logM + logN)
        Space: O(logM + logN)
        """
        if not matrix:
            return False
        if len(matrix) == 1:
            return self.searchArr(matrix[0], target)

        cur_row = math.floor((len(matrix) - 1) / 2)
        if matrix[cur_row][0] > target:
            return self.searchMatrix(matrix[:cur_row], target)
        if matrix[cur_row][-1] < target:
            return self.searchMatrix(matrix[cur_row+1:], target)
        return self.searchArr(matrix[cur_row], target)

    def searchArr(self, arr: List[int], target: int) -> bool:
        if not arr:
            return False
        if len(arr) == 1:
            return arr[0] == target

        cur = math.floor((len(arr)-1) / 2)
        if arr[cur] == target:
            return True
        if arr[cur] > target:
            return self.searchArr(arr[:cur], target)
        return self.searchArr(arr[cur+1:], target)

    def search(self, nums: List[int], target: int) -> int:
        """
        An int array nums sorted in ascending order (with distinct values).
        nums is possibly rotated at an unknown pivot index k (1 <= k < nums.length) such that 
        the resulting array is [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]].
        Find the index of target if it is in nums, or -1 if it is not in nums.
        e.x. nums = [4,5,6,7,0,1,2], target = 0 -> 4

        Time: O(logN)
        Space: O(1)
        """
        if len(nums) == 1:
            if nums[0] == target:
                return 0
            return -1

        start = 0
        end = len(nums) - 1

        while start <= end:
            mid = math.floor((start + end) / 2)
            if nums[mid] == target:
                return mid
            # pivot not in first half
            if nums[start] < nums[mid]:
                # target in first half
                if mid > start and nums[start] <= target <= nums[mid-1]:
                    end = mid - 1
                else:
                    start = mid + 1
            else:  # pivot in first half
                # target in second half
                if mid < end and nums[mid+1] <= target <= nums[end]:
                    start = mid + 1
                else:
                    end = mid - 1

        return -1

    def search(self, nums: List[int], target: int) -> int:
        """
        (RECURSIVE)

        Time: O(logN)
        Space: O(logN)
        """
        if not nums:
            return -1
        if len(nums) == 1:
            if nums[0] == target:
                return 0
            return -1

        mid = math.floor((len(nums) - 1) / 2)
        if nums[mid] == target:
            return mid

        # pivot not in first half but target in first half
        # or pivot in first half and target in first half (not in second half)
        if (nums[0] < nums[mid] and mid > 0 and nums[0] <= target <= nums[mid-1] or
            nums[0] > nums[mid] and mid < len(nums) - 1 and not (
                nums[mid+1] <= target <= nums[len(nums) - 1])):
            return self.search(nums[:mid], target)
        else:
            sec_half_idx = self.search(nums[mid+1:], target)
            return mid + 1 + sec_half_idx if sec_half_idx != -1 else -1

    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        """
        (TODO)
        2 sorted arrays nums1 and nums2 of size m and n, return the median of the two sorted arrays.
        e.x. nums1 = [1,2], nums2 = [3,4] -> 2.50000

        Time: O()
        Space: O()
        """
        # medium is either q+1_th smallest number or (q_th + q+1_th)/2
        q, r = divmod(len(nums1) + len(nums2), 2)
        return 0.0
