# Algorithms
1. Binary Search (iterative and recursive)
## iterative
* Make sure the items are sorted already
* Time complexity: O(Log N)
```java
public int binarySearch(int[] nums, int target) {
    int left = 0; right = nums.length - 1;
    while (left <= right) {
        int mid = (right + left / 2);
        if (nums[mid] < target) {
            right = mid - 1;
        } else if (nums[mid] > target) {
            left = mid + 1;
        } else {
            return mid;
        }
    }
    return -1;
}
```

## recursive
```java
public int binarySearch(int[] nums, int target, int left, int right) {
    if (left > right) return -1;

    int mid = (right + left / 2);
    if (nums[mid] < target) {
        right = mid - 1;
        return binarySearch(nums, target, left, right);
    } else if (nums[mid] > target) {
        left = mid + 1;
        return binarySearch(nums, target, left, right);
    } else {
        return mid;
    }
}
```

2. Nearest neighbor binary search
```java
public int binarySearch(int[] nums, int target) {
    int left = 0; right = nums.length - 1;
    int nearest = -1;
    while (left <= right) {
        if (left == right) return left;
        int mid = (right + left / 2);
        if (nums[mid] < target) {
            right = mid - 1;
        } else if (nums[mid] > target) {
            left = mid + 1;
        } else {
            return mid;
        }
    }
    
    if (left < nums.length && nums[left] - target > target - nums[right]) {
        nearest = left;
    } else if (right >= 0 && target - nums[right] < nums[left] - target) {
        nearest = right;
    } else if (left == nums.length) {
        nearest = nums.length - 1;
    } else if (right < 0) {
        nearest = 0;
    }
    
    return nearest;
}
```

* if the target number does not exist in the array, it's assumed that the last element evaluated, meaning the item where left == right, is one of the closest element. 

* When the loop is over, left > right.

* if left or right is out of bounds, the index 0 or nums.length - 1 needs to be returned, because it means the target number is less than the least number or the target number is greater than the greatest element.

# Practice Problems
1. https://leetcode.com/problems/find-k-closest-elements

* Time Complexity: O(N)
* Space Complexity: O(1)
```java
public List<Integer> findClosestElements(int[] arr, int k, int target) {
    if (k >= arr.length) {
        return Arrays.stream(arr).boxed().collect(Collectors.toList());
    }
    int left = 0;
    int right = arr.length - 1;
    while (right - left + 1 != k) {
        int mid = right + left / 2;
        if (target - arr[left] > arr[right] - target) {
            left++;
        } else {
            right--;
        }
    }
    List<Integer> ans = new ArrayList<>();
    while (left <= right) {
        ans.add(arr[left]);
        left++;
    }
    return ans;
}
```

* The purpose is to find the best window, so you need to compare the left and right not the mid, meaning the if statement needs to be like this `if (target - arr[left] > arr[right] - target) {`


2. https://leetcode.com/problems/search-a-2d-matrix/

* Time complexity: O(N) the size of matrix
* Space complexity: O(1)

* The key of this algorithm is to convert the mid value into row and column and continue the search as if it's a long array

```java
[0, 1, 2, 3, 4, 5, 6, 7]
[7, 8, 9, 10, 11, 12, 13, 14]
public boolean searchMatrix(int[][] matrix, int target) {
    int left = 0;
    int rowCount = matrix.length;
    int colCount = matrix[0].length;
    int right = rowCount * colCount - 1;
    
    while (left <= right) {
        int mid = (left + right) / 2;
        // translate
        int row = mid / colCount; // 11/7 = 1
        int col = mid % colCount;
        
        if (matrix[row][col] < target) {
            left = mid + 1;
        } else if (matrix[row][col] > target) {
            right = mid - 1;
        } else {
            return true;
        }
    }
    
    return false;
}
```

3. https://leetcode.com/problems/search-in-rotated-sorted-array/

* Time Complexity: O(log N)
* Space Complexity: O(1)

* The key is you have to check for the rotation area first. If the left is sorted, then see if the target belongs to the left. 
```java
public int search(int[] nums, int target) {
    int left = 0;
    int right = nums.length - 1;
    while (left <= right) {
        int mid = (left + right) / 2;
        if (nums[mid] == target) return mid;
        
        if (nums[mid] < nums[right]) { // check for the rotation first
            // mid to right is sorted
            if (nums[mid] < target && target <= nums[right]) { // mid < target <= right
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        } else { // if (nums[left] < nums[mid])
            // left to mid is sorted
            if (nums[mid] > target && target >= nums[left]) { // left <= target < mid
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
    }
    return -1;
}
```


4. https://leetcode.com/problems/median-of-two-sorted-arrays/

