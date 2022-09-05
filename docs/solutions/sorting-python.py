import collections
import math
import random
from typing import Callable, List, Tuple


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

    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        """
        Array of points [xi, yi] and an integer k, return the k closest points to the origin (0, 0).

        Time: O(N)
        Space: O(1)
        """
        # quick-select first
        random.shuffle(points)

        start = 0
        end = len(points) - 1

        while start < end + 1:
            partition_index = self.partitionPoints(points, start, end)
            if points[partition_index] == points[k-1]:
                return points[:k]
            elif partition_index > k - 1:
                end = partition_index - 1
            else:
                start = partition_index + 1

    def partitionPoints(self, points: List[List[int]], low: int, high: int) -> int:
        """
        Helper function to partition the points in place (2-way) according to distance to origin. 
        Partition value is points[low]'s distance to origin.

        Args:
            points (List[List[int]]): list to partition, will be changed in-place
            low (int): start index of the subarray to partition
            high (int): end index of the subarray to partition

        Returns:
            int: the partition point index where points[low: p] closer to origin than -> 
                 points[p] closer or same to origin than -> points[p: high+1]

        Time: O(N)
        Space: O(1)
        """
        start = low + 1
        end = high
        partition_dist = math.pow(
            points[low][0], 2) + math.pow(points[low][1], 2)

        while start < end + 1:
            if math.pow(points[start][0], 2) + math.pow(points[start][1], 2) < partition_dist:
                start += 1
                continue
            if math.pow(points[end][0], 2) + math.pow(points[end][1], 2) >= partition_dist:
                end -= 1
                continue

            points[start], points[end] = points[end], points[start]

        points[low], points[start-1] = points[start-1], points[low]
        return start-1

    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        """
        Array of points [xi, yi] and an integer k, return the k closest points to the origin (0, 0).
        Use partition 3-way.

        Time: O(N)
        Space: O(1)
        """
        random.shuffle(points)
        start = 0
        end = len(points) - 1

        while start < end + 1:
            p_smaller, p_greater = self.partitionPointsThreeWays(
                points, start, end)
            if p_smaller <= k - 1 < p_greater:
                return points[:k]
            if k - 1 < p_smaller:
                end = p_smaller - 1
            else:
                start = p_greater

    def partitionPointsThreeWays(self, points: List[List[int]], low: int, high: int) -> List[int]:
        """
        Helper function to partition the points in place (3-way) according to distance to origin. 
        Partition value is points[low]'s distance to origin.

        Args:
            points (List[List[int]]): list to partition, will be changed in-place
            low (int): start index of the subarray to partition
            high (int): end index of the subarray to partition

        Returns:
            List[int]: the partition points indices where points[low: p1] closer to origin than -> 
                       points[p1: p2] closer to origin than -> points[p2: high+1]

        Time: O(N)
        Space: O(1)
        """
        p_smaller = p_smaller_equal = low + 1
        p_greater = high
        p_dist = math.pow(points[low][0], 2) + math.pow(points[low][1], 2)

        while p_smaller_equal < p_greater + 1:
            # move p_smaller_equal and compare with p_dist
            cur_dist = math.pow(
                points[p_smaller_equal][0], 2) + math.pow(points[p_smaller_equal][1], 2)

            if cur_dist < p_dist:
                # if < p_dist, swap with p_smaller and move both p_smaller/p_smaller_equal forward
                # points[low+1: p_smaller] < p_dist, points[p_smaller: p_smaller_equal] == p_dist
                points[p_smaller], points[p_smaller_equal] = points[p_smaller_equal], points[p_smaller]
                p_smaller += 1
                p_smaller_equal += 1
            elif cur_dist == p_dist:
                # if == p_dist, move p_smaller_equal forward, points[p_smaller: p_smaller_equal] == p_dist
                p_smaller_equal += 1
            else:
                # if > p_dist, swap with p_greater and move p_greater backward
                # points[p_greater+1:] > p_dist, value of points[p_smaller_equal] after swap unknown
                points[p_smaller_equal], points[p_greater] = points[p_greater], points[p_smaller_equal]
                p_greater -= 1

        points[low], points[p_smaller-1] = points[p_smaller-1], points[low]
        return [p_smaller-1, p_smaller_equal]

    def topKFrequent(self, words: List[str], k: int) -> List[str]:
        """
        Given an array of strings words and an integer k, return the k most frequent strings.

        Returns:
            List[str]: k most frequent strings sorted by the frequency from highest to lowest. 
                       Sort the words with the same frequency by their lexicographical order.

        Time: O(KlogK + N)
        Space: O(logK)
        """
        word_map = collections.Counter(words)
        word_freq = [(word, freq) for word, freq in word_map.items()]

        if k >= len(word_map):
            self.sortWords(word_freq, 0, len(word_freq)-1)
            return [wf[0] for wf in word_freq]

        start = 0
        end = len(word_freq) - 1
        while start < end + 1:
            p_index = self.partitionWords(word_freq, start, end)
            if p_index == k-1:
                self.sortWords(word_freq, 0, k-1)
                return [wf[0] for wf in word_freq[:k]]
            if p_index < k - 1:
                start = p_index + 1
            else:
                end = p_index - 1

    def compare(self, str_freq_1: Tuple[str, int], str_freq_2: Tuple[str, int]) -> int:
        """
        Compare function of (string, frequence) tuples. Won't have duplicate values.
        Return: -1 if str_freq_1 should be before str_freq_2
                 1 otherwise
        """
        # more frequent word should be before less frequent
        if str_freq_1[1] < str_freq_2[1]:
            return 1
        if str_freq_1[1] > str_freq_2[1]:
            return -1
        # if same frequent, smaller lexi order word should be before
        if str_freq_1[0] < str_freq_2[0]:
            return -1
        return 1  # impossible to be equal

    def sortWords(self, words: List[Tuple[str, int]], low: int, high: int):
        """
        Sort words and update array in place using quicksort. Sort by frequence and lexi order.

        Args:
            words (List[Tuple[str, int]]): list of (string, frequence) that containing the part to sort
            low (int): start index of subarray to sort
            high (int): end index of subarray to sort

        Time: O(NlogN)
        Space: O(logN)
        """

        # use insertion sort if array is short
        if high - low + 1 < 5:
            for i in range(low, high):
                for j in range(i+1, high+1):
                    if self.compare(words[i], words[j]) == 1:
                        words[i], words[j] = words[j], words[i]
            return

        # quicksort
        p_index = self.partitionWords(words, low, high)
        if p_index < 2:
            self.sortWords(words, p_index+1, high)
        else:
            self.sortWords(words, low, p_index-1)
            self.sortWords(words, p_index+1, high)

    def partitionWords(self, words: List[Tuple[str, int]], low: int, high: int) -> int:
        """
        Helper function to partition the words in place (2-way) according to frequency and lexi order. 
        Partition value is words[low]. In this case there won't be dup values so no need to do 3-way.

        Args:
            words (List[Tuple[str, int]]): list to partition, will be changed in-place
            low (int): start index of the subarray to partition
            high (int): end index of the subarray to partition

        Returns:
            List[int]: partition point index where words[low: p] frequency/lexi order < words[p: high+1]

        Time: O(N)
        Space: O(1)
        """
        start = low + 1
        end = high
        p_val = words[low]

        while start < end + 1:
            if self.compare(words[start], p_val) == -1:
                start += 1
                continue
            if self.compare(words[end], p_val) == 1:
                end -= 1
                continue
            words[start], words[end] = words[end], words[start]

        words[start-1], words[low] = words[low], words[start-1]
        return start-1

    def quickSort(self, array_obj: List[object], low: int, high: int, compare_func: Callable):
        """
        General quicksort function to help with later solutions. Sort in place.

        Args:
            array_obj (List[object]): array that contains the part to sort
            low (int): start index of subarray to sort
            high (int): end index of subarray to sort
            compare_func (Callable): compare function to use, -1/0/1 as </==/>

        Time: O(NlogN)
        Space: O(logN)
        """
        # insertion sort if length is < 5
        if high - low + 1 < 5:
            for i in range(low, high):
                for j in range(i+1, high+1):
                    if compare_func(array_obj[i], array_obj[j]) > 0:
                        array_obj[i], array_obj[j] = array_obj[j], array_obj[i]
            return

        # quicksort
        partition_index = self.partitionObj(array_obj, low, high, compare_func)
        if partition_index < 2:
            self.quickSort(array_obj, partition_index+1, high, compare_func)
        else:
            self.quickSort(array_obj, low, partition_index-1, compare_func)
            self.quickSort(array_obj, partition_index+1, high, compare_func)

    def partitionObj(self, array_obj: List[object], low: int, high: int, compare_func: Callable) -> int:
        """
        General partion function (2-way) to help with later solutions. Partition in place.

        Args:
            array_obj (List[object]): array that contains the part to partition
            low (int): start index of subarray to partition
            high (int): end index of subarray to partition
            compare_func (Callable): compare function to use, -1/0/1 as </==/>

        Returns:
            int: partition point index where array_obj[low: p] < array_obj[p] <= array_obj[p+1:high+1]

        Time: O(N)
        Space: O(1)
        """
        start = low + 1
        end = high
        p_val = array_obj[low]

        while start < end + 1:
            if compare_func(array_obj[start], p_val) < 0:
                start += 1
                continue
            if compare_func(array_obj[end], p_val) >= 0:
                end -= 1
                continue
            array_obj[start], array_obj[end] = array_obj[end], array_obj[start]

        array_obj[low], array_obj[start-1] = array_obj[start-1], array_obj[low]
        return start-1

    def compareIntervals(self, interval_1: List[int], interval_2: List[int]) -> int:
        if interval_1[0] < interval_2[0]:
            return -1
        if interval_1[0] == interval_2[0]:
            return 0
        return 1

    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        """
        Merge all overlapping intervals.

        Args:
            intervals (List[List[int]]): array of intervals where intervals[i] = [starti, endi]

        Returns:
            List[List[int]]: array of non-overlapping intervals that cover all the intervals in the input

        Time: O(NlogN)
        Space: O(logN)
        """
        # sort on start
        random.shuffle(intervals)
        self.quickSort(intervals, 0, len(intervals)-1, self.compareIntervals)

        # check end and if can merge
        i = 1
        while i < len(intervals):
            # [start_i-1, end_i-1], [start_i, end_i]: start_i <= end_i-1
            if intervals[i][0] <= intervals[i-1][1]:
                intervals[i-1][1] = max(intervals[i-1][1], intervals[i][1])
                intervals.pop(i)
            else:
                i += 1

        return intervals

    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        """
        Given an array of non-overlapping intervals sorted in ascending order by starti and a new interval.
        Insert newInterval into intervals, still sorted in ascending order by starti and still non-overlapping.

        Args:
            intervals (List[List[int]]): intervals[i] = [starti, endi] sorted by starti
            newInterval (List[int]): [start, end] that represents the start and end of another interval

        Returns:
            List[List[int]]: new intervals with inserted interval and merged if necessary

        Time: O(N)
        Space: O(1) - not counting the returning list
        """
        new_intervals = []
        inserted = False

        def insertNewInterval(new_itv: List[int]):
            # [start_-1, end_-1], [start_ni, end_ni]: start_ni <= end_-1 ?
            if not new_intervals or new_itv[0] > new_intervals[-1][1]:
                new_intervals.append(new_itv)
            elif new_itv[1] > new_intervals[-1][1]:
                new_intervals[-1][1] = new_itv[1]

        for itv in intervals:
            # find appropriate place to insert newInterval and merge potentially
            if not inserted and newInterval[0] <= itv[0]:
                insertNewInterval(newInterval)
                inserted = True

            # insert itv and check if needs to merge
            insertNewInterval(itv)

        if not inserted:
            insertNewInterval(newInterval)

        return new_intervals

    def canAttendMeetings(self, intervals: List[List[int]]) -> bool:
        """
        Given an array of meeting time intervals, determine if a person could attend all meetings.

        Args:
            intervals (List[List[int]]): meeting intervals where intervals[i] = [starti, endi]

        Returns:
            bool: whether can attend all meetings

        Time: O(NlogN)
        Space: O(logN)
        """
        # sort on start
        random.shuffle(intervals)
        self.quickSort(intervals, 0, len(intervals)-1, self.compareIntervals)

        # check if there is any overlap
        for i in range(1, len(intervals)):
            if intervals[i][0] < intervals[i-1][1]:
                return False

        return True

    def minMeetingRooms(self, intervals: List[List[int]]) -> int:
        """
        Given an array of meeting time intervals, return the minimum number of conference rooms required.

        Args:
            intervals (List[List[int]]): intervals where intervals[i] = [starti, endi]

        Returns:
            int: minimum number of conference rooms required

        Time: O(NlogN + num of rooms needed ^2) - worst O(N^2) 
        Space: O(N)
        """
        if len(intervals) < 2:
            return len(intervals)

        # sort on start
        random.shuffle(intervals)
        self.quickSort(intervals, 0, len(intervals)-1, self.compareIntervals)

        # check overlap against each meeting room
        room_itv = []
        for itv in intervals:
            if not room_itv:
                room_itv.append([itv])
                continue

            can_fit_in = False
            for room in room_itv:
                if room[-1][1] <= itv[0]:
                    can_fit_in = True
                    room.append(itv)
                    break
            if not can_fit_in:
                room_itv.append([itv])

        return len(room_itv)

    def minMeetingRooms(self, intervals: List[List[int]]) -> int:
        """
        (Another good algorithm from solution)
        Given an array of meeting time intervals, return the minimum number of conference rooms required.

        Args:
            intervals (List[List[int]]): intervals where intervals[i] = [starti, endi]

        Returns:
            int: minimum number of conference rooms required

        Time: O(NlogN)
        Space: O(N)
        """
        # [[0,30],[5,10],[15,20],[10,15]]
        start_times = sorted([itv[0] for itv in intervals])  # 0, 5, 10, 15
        end_times = sorted([itv[1] for itv in intervals])  # 10, 15, 20, 30
        num_room_in_use = 0
        num_room_needed = 0

        s = e = 0
        while s < len(start_times):
            # no meeting has ended, have to occupy a new room
            if start_times[s] < end_times[e]:
                num_room_in_use += 1
                num_room_needed = max(num_room_needed, num_room_in_use)
                s += 1
            if s == len(start_times):
                break

            # some meetings will end before next meeting starts, wait for them to end
            if end_times[e] <= start_times[s]:
                num_room_in_use -= 1
                e += 1

        return num_room_needed


class MyCalendar:
    """
    A calendar object that can add a new event if adding the event will not cause a double booking.

    Time: O(NlogN) - for booking N events
    Space: O(N)
    """

    def __init__(self):
        self.bookings = []

    def book(self, start: int, end: int) -> bool:
        """
        Time: O(logN)
        Space: O(1)
        """
        # if no bookings, can definitely book
        if not self.bookings:
            self.bookings.append([start, end])
            return True

        # if start and end before first booking's start, can book
        if start <= self.bookings[0][0]:
            if end <= self.bookings[0][0]:
                self.bookings.insert(0, [start, end])
                return True
            return False

        # if start and end later than last booking's end, can book
        if end >= self.bookings[-1][1]:
            if start >= self.bookings[-1][1]:
                self.bookings.append([start, end])
                return True
            return False

        # if currently only 1 booking and not before or after it, cannot book
        if len(self.bookings) == 1:
            return False

        # find a point where start_i-1 <= start <= start_i, if cannot find, cannot insert
        # (have checked head and tail insert already above)
        left = 0
        right = len(self.bookings) - 1
        cur_index = 0
        while left < right + 1:
            cur_index = math.floor((left + right) / 2)
            if cur_index == 0:
                left = cur_index + 1
                continue
            prev_start = self.bookings[cur_index-1][0]
            cur_start = self.bookings[cur_index][0]
            if prev_start <= start <= cur_start:
                break
            if start < prev_start:
                right = cur_index - 1
                continue
            left = cur_index + 1

        if right < left:
            return False

        # check if can insert at this index
        if start >= self.bookings[cur_index-1][1] and end <= self.bookings[cur_index][0]:
            self.bookings.insert(cur_index, [start, end])
            return True
        return False


class MyCalendarTwo:
    """
    A calendar object that can add a new event if adding the event will not cause a triple booking.

    Time: O(NlogN) - for booking N events
    Space: O(N)
    """

    def __init__(self):
        self.bookings = []

    def book(self, start: int, end: int) -> bool:
        """
        Time: O(logN) - worst O(N)
        Space: O(1)
        """
        # not overlapping with current bookings at all
        if not self.bookings or start >= self.bookings[-1][0][1]:
            self.bookings.append([[start, end], 1])
            return True

        # find the insert index, e_i-1 <= s < e_i
        left = 0
        right = len(self.bookings) - 1
        insert_index = 0
        while left < right + 1:
            insert_index = math.floor((left + right) / 2)

            if insert_index == 0:
                # start < end_0
                if start < self.bookings[insert_index][0][1]:
                    break
                left = insert_index + 1
                continue

            # e_i-1 <= s < e_i
            if self.bookings[insert_index-1][0][1] <= start < self.bookings[insert_index][0][1]:
                break
            # s >= e_i
            if start >= self.bookings[insert_index][0][1]:
                left = insert_index + 1
                continue
            # s < e_i-1
            if start < self.bookings[insert_index-1][0][1]:
                right = insert_index - 1
                continue

        # check if can insert
        cur_index = insert_index
        while cur_index < len(self.bookings) and self.bookings[cur_index][0][0] < end:
            if self.bookings[cur_index][1] == 2:
                return False
            cur_index += 1

        # insert at insert_index
        self.insertAt(insert_index, start, end)
        return True

    def insertAt(self, index: int, start: int, end: int):
        """
        Helper function to insert an interval [start, end] at index, already checked can insert.

        Time: O(N) in worst case but it can only happen one time
        Space: O(1)
        """
        if start > end - 1:
            return

        if index == len(self.bookings):
            self.bookings.append([[start, end], 1])
            return

        cur_start, cur_end = self.bookings[index][0]

        # can insert before or merge with current interval
        if end < cur_start or end == cur_start and self.bookings[index][1] == 2:
            self.bookings.insert(index, [[start, end], 1])
            return
        if end == cur_start:
            self.bookings[index][0][0] = start
            return

        # handle start cases where start != cur_start
        if cur_start < start:
            self.bookings[index][0][0] = start
            self.bookings.insert(index, [[cur_start, start], 1])
            self.insertAt(index + 1, start, end)
            return
        if cur_start > start:
            self.bookings.insert(index, [[start, cur_start], 1])
            self.insertAt(index + 1, cur_start, end)
            return

        # handle end cases where end != cur_end
        if cur_end > end:
            self.bookings[index][0][0] = end
            self.bookings.insert(index, [[start, end], 2])
            return
        if cur_end < end:
            self.bookings[index][1] = 2
            self.insertAt(index + 1, cur_end, end)
            return

        # start == cur_start, end == cur_end
        self.bookings[index][1] = 2


class MyCalendarThree:
    """
    A calendar object that can add a new event and return the maximum k-booking between previous events.

    Time: O(NlogN) - worst O(N^2) - for booking N events
    Space: O(N)
    """

    def __init__(self):
        self.bookings = []
        self.max_overbook = 0

    def book(self, start: int, end: int) -> int:
        """
        Time: O(logN) - worst O(N)
        Space: O(1)
        """
        # not overlapping with current bookings at all
        if not self.bookings:
            self.bookings.append([[start, end], 1])
            self.max_overbook = 1
            return 1
        if start > self.bookings[-1][0][1]:
            self.bookings.append([[start, end], 1])
            return self.max_overbook
        if start == self.bookings[-1][0][1]:
            if self.bookings[-1][1] == 1:
                self.bookings[-1][0][1] = end
            else:
                self.bookings.append([[start, end], 1])
            return self.max_overbook

        # find the insert index, e_i-1 <= s < e_i
        left = 0
        right = len(self.bookings) - 1
        insert_index = 0
        while left < right + 1:
            insert_index = math.floor((left + right) / 2)

            if insert_index == 0:
                # start < end_0
                if start < self.bookings[insert_index][0][1]:
                    break
                left = insert_index + 1
                continue

            # e_i-1 <= s < e_i
            if self.bookings[insert_index-1][0][1] <= start < self.bookings[insert_index][0][1]:
                break
            # s >= e_i
            if start >= self.bookings[insert_index][0][1]:
                left = insert_index + 1
                continue
            # s < e_i-1
            if start < self.bookings[insert_index-1][0][1]:
                right = insert_index - 1
                continue

        # insert at insert_index and update max_overbook
        self.insertAt(insert_index, start, end)
        return self.max_overbook

    def insertAt(self, index: int, start: int, end: int):
        """
        Helper function to insert an interval [start, end] at index, and update max_overbook.

        Time: O(N) in worst case
        Space: O(N) in worst case
        """
        if start > end - 1:
            return

        if index == len(self.bookings):
            self.bookings.append([[start, end], 1])
            return

        cur_start, cur_end = self.bookings[index][0]

        # can insert before or merge with current interval
        if end < cur_start or end == cur_start and self.bookings[index][1] > 1:
            self.bookings.insert(index, [[start, end], 1])
            return
        if end == cur_start:
            self.bookings[index][0][0] = start
            return

        # handle start cases where start != cur_start
        if cur_start < start:
            self.bookings[index][0][0] = start
            self.bookings.insert(
                index, [[cur_start, start], self.bookings[index][1]])
            self.insertAt(index + 1, start, end)
            return
        if cur_start > start:
            self.bookings.insert(index, [[start, cur_start], 1])
            self.insertAt(index + 1, cur_start, end)
            return

        # handle end cases where end != cur_end
        if cur_end > end:
            self.bookings[index][0][0] = end
            self.bookings.insert(
                index, [[start, end], self.bookings[index][1] + 1])
            self.max_overbook = max(self.max_overbook, self.bookings[index][1])
            return
        if cur_end < end:
            self.bookings[index][1] += 1
            self.max_overbook = max(self.max_overbook, self.bookings[index][1])
            self.insertAt(index + 1, cur_end, end)
            return

        # start == cur_start, end == cur_end
        self.bookings[index][1] += 1
        self.max_overbook = max(self.max_overbook, self.bookings[index][1])


class Solution:

    def carPooling(self, trips: List[List[int]], capacity: int) -> bool:
        """
        A car with capacity empty seats.
        An array trips where trips[i] = [numPassengersi, fromi, toi] indicates that 
        the ith trip has numPassengersi passengers and the locations are fromi and toi respectively.

        Check if it is possible to pick up and drop off all passengers for all the given trips.
        e.x. trips = [[2,1,5],[3,3,7]], capacity = 4 -> false

        Time: O(NlogN) - worst O(N^2)
        Space: O(N)
        """
        pass_ints = []  # keep track of number of passengers on the car at each interval

        for trip in trips:
            num_passengers, start, end = trip
            if num_passengers > capacity:
                return False

            if not pass_ints:
                self.insertAt(pass_ints, trip, 0)
                continue

            # binary search to insert trip into valid trips potentially
            # find index such that end_i-1 <= start < end_i
            left_idx = 0
            right_idx = len(pass_ints) - 1
            insert_idx = 0
            while left_idx < right_idx + 1:
                insert_idx = math.floor((left_idx + right_idx) / 2)
                cur_end = pass_ints[insert_idx][2]
                if start < cur_end and insert_idx == 0:
                    break
                prev_end = pass_ints[insert_idx - 1][2]
                if prev_end <= start < cur_end:
                    break
                if cur_end <= start:
                    left_idx = insert_idx + 1
                    continue
                # prev_end > start
                right_idx = insert_idx - 1
            if left_idx > right_idx:
                insert_idx = left_idx

            # check if overlap with next intervals that starts before end
            cur_idx = insert_idx
            while cur_idx < len(pass_ints):
                next_num_passengers, next_start, _ = pass_ints[cur_idx]
                if end <= next_start:
                    break
                if num_passengers + next_num_passengers > capacity:
                    return False
                cur_idx += 1

            # insert this trip into pass_ints and update
            self.insertAt(pass_ints, trip, insert_idx)

        return True

    def insertAt(self, pass_ints: List[List[int]], trip: List[int], insert_index: int):
        """
        Helper function to insert a trip into pass_ints, may need to split and merge.

        Time: O(N) in worst case
        Space: O(N) in worst case
        """
        num_pass, start, end = trip
        if end <= start:
            return

        if not pass_ints:
            pass_ints.append(trip)
            return

        # can append or merge with last interval
        last_num_pass, _, last_end = pass_ints[-1]
        if insert_index == len(pass_ints) and (start > last_end or start == last_end and num_pass != last_num_pass):
            pass_ints.append(trip)
            return
        if insert_index == len(pass_ints) and start == last_end and num_pass == last_num_pass:
            pass_ints[-1][2] = end
            return

        # can insert at front or merge with first interval
        first_num_pass, first_start, _ = pass_ints[0]
        if insert_index == 0 and end < first_start:
            pass_ints.insert(0, trip)
            return
        if insert_index == 0 and end == first_start and num_pass == first_num_pass:
            pass_ints[0][1] = start
            return

        # handle cases where start != cur_start
        cur_num_pass, cur_start, cur_end = pass_ints[insert_index]
        if start < cur_start:
            pass_ints.insert(insert_index, [num_pass, start, cur_start])
            self.insertAt(
                pass_ints, [num_pass, cur_start, end], insert_index + 1)
            return
        if start > cur_start:
            pass_ints[insert_index][1] = start
            pass_ints.insert(insert_index, [cur_num_pass, cur_start, start])
            self.insertAt(pass_ints, trip, insert_index + 1)
            return

        # handle cases where end != cur_end
        if end > cur_end:
            pass_ints[insert_index][0] += num_pass
            self.insertAt(
                pass_ints, [num_pass, cur_end, end], insert_index + 1)
            return
        if end < cur_end:
            pass_ints[insert_index][1] = end
            pass_ints.insert(
                insert_index, [num_pass + cur_num_pass, start, end])
            return

        # start == cur_start, end == cur_end
        pass_ints[insert_index][0] += num_pass

    def carPooling(self, trips: List[List[int]], capacity: int) -> bool:
        """
        Another algorithm

        Time: O(NlogN)
        Space: O(N)
        """
        # sort with start time
        trips.sort(key=lambda t: t[1])
        start_load = [(t[1], t[0]) for t in trips]
        # print(start_load)

        # sort with end time
        trips.sort(key=lambda t: t[2])
        end_load = [(t[2], t[0]) for t in trips]
        # print(end_load)

        start_idx = end_idx = 0
        cur_load = 0

        while start_idx < len(start_load):
            cur_start, cur_add_load = start_load[start_idx]
            cur_end, cur_del_load = end_load[end_idx]

            if cur_start < cur_end:
                cur_load += cur_add_load
                if cur_load > capacity:
                    return False
                start_idx += 1
                continue

            if cur_end <= cur_start:
                cur_load -= cur_del_load
                end_idx += 1

        return True

    def carPooling(self, trips: List[List[int]], capacity: int) -> bool:
        """
        Another algorithm from discussion section.

        Time: O(N + max distance)
        Space: O(N)
        """
        start_load = {}
        end_load = {}
        max_distance = 0

        # take record of at which location we have how many passengers aboard and leave
        # and the max distance we have travelled
        for num_passengers, start, end in trips:
            if num_passengers > capacity:
                return False

            if start not in start_load:
                start_load[start] = 0
            start_load[start] += num_passengers

            if end not in end_load:
                end_load[end] = 0
            end_load[end] += num_passengers

            max_distance = max(max_distance, end)

        # at each location, unload leaving passengers and load new passengers
        # and check load vs. capacity
        cur_load = 0
        for cur_loc in range(max_distance):
            # anyone leaves?
            if cur_loc in end_load:
                cur_load -= end_load[cur_loc]
            # anyone onboards?
            if cur_loc in start_load:
                cur_load += start_load[cur_loc]
                if cur_load > capacity:
                    return False

        return True


class Swapping:
    def minSwapping(self, nums: List[int]) -> int:
        """
        (Algorithm from the GeeksForGeeks page)
        Given an array of n distinct elements, find the minimum number of swaps required to sort the array.

        Time: O(NlogN)
        Space: O(N)
        """
        # use a graph cycle to determine how many swaps we need
        # e.x.
        #   |---------->|                 |---------->|
        #   5  6  2  3  1  4           5  6  2  3  1  4
        #   |<----------|                 |<-|<-|<----|
        # for each cycle, number of swaps needed is the number of edges - 1

        min_swaps = 0
        num_idx = {}  # correct idx of each num
        old_nums = nums.copy()  # space O(N)
        nums.sort()  # O(NlogN)
        for i, num in enumerate(nums):
            num_idx[num] = i
        # print(num_idx)  # {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}

        cur_circle = []
        res_nums = set(nums)

        for num in old_nums:  # O(N)
            if num not in res_nums:
                continue

            # form the circle
            start_num = cur_num = num
            while True:
                cur_circle.append(cur_num)
                res_nums.remove(cur_num)
                # find the number who is currently at the index cur_num is supposed to be at
                cur_num = old_nums[num_idx[cur_num]]
                if cur_num == start_num:
                    min_swaps += len(cur_circle) - 1
                    cur_circle = []
                    break

        return min_swaps

    def minSwappingNoGraph(self, nums: List[int]) -> int:
        """
        (Algorithm from the GeeksForGeeks page: no graph, keep swapping to keep prefix of array sorted.
         Intuitively I think it's correct but I haven't tried to prove it.)
        Given an array of n distinct elements, find the minimum number of swaps required to sort the array.

        Time: O(NlogN)
        Space: O(N)
        """
        min_swap = 0
        old_nums = nums.copy()
        old_num_idx = {}
        for i, num in enumerate(old_nums):
            old_num_idx[num] = i

        nums.sort()
        # keep [:i] always sorted
        for i, num in enumerate(old_nums):
            if num == nums[i]:
                continue
            # swap i and correct num that should be at i
            correct_num_i = old_num_idx[nums[i]]
            old_nums[i], old_nums[correct_num_i] = old_nums[correct_num_i], old_nums[i]
            min_swap += 1

        return min_swap


if __name__ == "__main__":
    s = Swapping()
    min_swap = s.minSwapping([5, 6, 2, 3, 1, 4])
    print(min_swap)
    min_swap = s.minSwapping([4, 3, 2, 1])
    print(min_swap)
    min_swap = s.minSwapping([1, 5, 4, 3, 2])
    print(min_swap)

    min_swap = s.minSwappingNoGraph([5, 6, 2, 3, 1, 4])
    print(min_swap)
    min_swap = s.minSwappingNoGraph([4, 3, 2, 1])
    print(min_swap)
    min_swap = s.minSwappingNoGraph([1, 5, 4, 3, 2])
    print(min_swap)
