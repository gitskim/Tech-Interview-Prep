# Backtracking
* https://leetcode.com/problems/letter-combinations-of-a-phone-number/

* Time complexity: O(4^N)
* Space complexity: O(4^N * N)

## preorder
```java
private String digits;
private Map<Character, String> map;
public List<String> letterCombinations(String digits) {
    this.digits = digits;
    this.map = new HashMap<>();
    map.put('2', "abc");
    map.put('3', "def");
    map.put('4', "ghi");
    map.put('5', "jkl");
    map.put('6', "mno");
    map.put('7', "pqrs");
    map.put('8', "tuv");
    map.put('9', "wxyz");
    List<String> ans = new ArrayList<>();
    if (digits.length() == 0) return ans;
    function(0, 0, ans, new StringBuilder());
    return ans;
}

private void function(int istart, int order, List<String> ans, StringBuilder sb) {
    if (order == digits.length()) {
        ans.add(sb.toString());
        return;
    }
    
    char bundle = digits.charAt(order);
    String comb = map.get(bundle);
    for (int i = istart; i < comb.length(); i++) {
        sb.append(comb.charAt(i));
        function(0, order + 1, ans, sb);
        sb.deleteCharAt(sb.length() - 1);
    }
}
```

## postorder
```java
private List<String> function(int order) {
    if (order == digits.length()) {
        return new ArrayList<>();
    }
    
    char bundle = digits.charAt(order);
    String comb = map.get(bundle);
    List<String> myWork = new ArrayList<>(); // used by every sibling
    
    for (int i = 0; i < comb.length(); i++) {
        List<String> childWork = function(order + 1);
        
        if (childWork.size() == 0) {
            myWork.add(comb.substring(i, i + 1));
        } else {
            for (String s : childWork) {
                myWork.add(comb.substring(i, i + 1) + s);
            }
        }
    }
    return myWork;
}
```
* https://leetcode.com/problems/subsets-ii/
* Time Complexity: O(2^N)
* Space Complexity: O(2^N * N)
## preorder
```java
public List<List<Integer>> subsetsWithDup(int[] nums) {
    Set<List<Integer>> ans = new HashSet<>();
    Arrays.sort(nums);
    preorder(nums, 0, ans, new ArrayList<>());
    return new ArrayList<>(ans);
}

private void preorder(int[] nums, int start, Set<List<Integer>> answer, List<Integer> interm) {
    answer.add(new ArrayList<>(interm));
    if (start == nums.length) {
        return;
    }
    
    for (int i = start; i < nums.length; i++) {
        interm.add(nums[i]);
        preorder(nums, i + 1, answer, interm);
        interm.remove(interm.size() - 1); // remove the current node before moving on to the next sibling
    }
}
```
## postorder

```java
private Set<List<Integer>> postorder(int[] nums, int start) {
    if (start == nums.length) {
        Set<List<Integer>> ans = new HashSet<>() {{
            add(new ArrayList<>());
        }};
        return ans;
    }
    
    Set<List<Integer>> childWork = postorder(nums, start + 1);
    Set<List<Integer>> curWork = new HashSet<>();
    for (List<Integer> list : childWork) {
        curWork.add(list);
        List<Integer> newList = new ArrayList<>(list);
        newList.add(nums[start]); // important - the order of which one you add first matters in set
        curWork.add(newList);
    }
    return curWork;
}
```
* https://leetcode.com/problems/combinations/
* https://leetcode.com/problems/combination-sum/
* https://leetcode.com/problems/palindrome-partitioning/
* https://leetcode.com/problems/permutations/
* https://leetcode.com/problems/permutations-ii/
* https://leetcode.com/problems/generate-parentheses/
