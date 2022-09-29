import math
from typing import List, Optional


class ListNode:
    """
    Definition for singly-linked list
    """

    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        (ITERATIVE)
        Given the head of a singly linked list, reverse the list, and return the reversed list.

        Time: O(N)
        Space: O(1)
        """
        prev_node = next_node = None
        cur_node = head

        while cur_node:
            # (prev_node)(cur_node)(next_node)
            # node_i-1    node_i -> node_i+1
            next_node = cur_node.next
            # (prev_node)(cur_node) (next_node)
            # node_i-1 <- node_i     node_i+1
            cur_node.next = prev_node
            #           (prev_node) (next_node)
            # node_i-1 <- node_i     node_i+1
            prev_node = cur_node
            #           (prev_node) (cur_node)
            # node_i-1 <- node_i     node_i+1
            cur_node = next_node

        return prev_node

    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        (RECURSIVE)

        Time: O(N)
        Space: O(N)
        """
        rev_head, rev_tail = self.reverseListHeadTail(head)
        if rev_tail:
            rev_tail.next = None
        return rev_head

    def reverseListHeadTail(self, head: Optional[ListNode]) -> List[Optional[ListNode]]:
        """
        Helper function to recursively reverse a list and return the head and tail of the reversed list.
        """
        if not head:
            return [None, None]
        if not head.next:
            return [head, head]

        rev_res_head, rev_res_tail = self.reverseListHeadTail(head.next)
        rev_res_tail.next = head
        return [rev_res_head, head]

    def reverseBetween(self, head: Optional[ListNode], left: int, right: int) -> Optional[ListNode]:
        """
        (ITERATIVE)
        Given the head of a singly linked list and two integers left and right where left <= right, 
        reverse the nodes of the list from left to right, and return the reversed list.
        e.x. head = [1,2,3,4,5], left = 2, right = 4 -> [1,4,3,2,5]

        Time: O(N)
        Space: O(1)
        """
        left_node = right_node = reversed_tail = reversed_head = None
        prev_node = next_node = None
        cur_node = head

        # go through nodes prior to left
        i = 1
        while i < left:
            prev_node = cur_node
            cur_node = cur_node.next
            i += 1

        # mark the left_node and reversed_tail
        left_node = prev_node
        reversed_tail = cur_node

        # go through nodes from left to right and reverse list
        while i <= right:
            next_node = cur_node.next
            cur_node.next = prev_node
            prev_node = cur_node
            cur_node = next_node
            i += 1

        # mark the right_node and reversed_head
        right_node = cur_node
        reversed_head = prev_node

        # connect left_node and right_node to the middle reversed list
        if left_node:
            left_node.next = reversed_head
        else:
            head = reversed_head
        reversed_tail.next = right_node

        return head

    def reverseBetween(self, head: Optional[ListNode], left: int, right: int) -> Optional[ListNode]:
        """
        (RECURSIVE)

        Time: O(N)
        Space: O(N)
        """
        left_node = right_node = head_to_rev = None
        prev_node = None
        cur_node = head

        # get left_node and head of the list to reverse
        i = 1
        while i < left:
            prev_node = cur_node
            cur_node = cur_node.next
            i += 1

        left_node = prev_node
        head_to_rev = cur_node

        # get right_node and break the tail of the list to reverse
        while i <= right:
            prev_node = cur_node
            cur_node = cur_node.next
            i += 1

        right_node = cur_node
        prev_node.next = None

        # reverse left to right
        rev_head, rev_tail = self.reverseListHeadTail(head_to_rev)

        # connect reversed head and tail to left and right node
        if left_node:
            left_node.next = rev_head
        else:
            head = rev_head
        rev_tail.next = right_node

        return head

    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        """
        Given the head of a linked list, remove the nth node from the END of the list and return its head.

        Time: O(N)
        Space: O(1)
        """
        gap = n
        prev_del_node = None
        del_node = cur_node = head

        while cur_node:
            cur_node = cur_node.next
            if gap >= 0:
                gap -= 1
            else:
                prev_del_node = del_node
                del_node = del_node.next

        if gap > 0:
            return head
        if not prev_del_node:
            return del_node.next
        prev_del_node.next = del_node.next
        return head

    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        """
        Given the head of a singly linked list, return true if it is a palindrome or false otherwise.

        Time: O(N)
        Space: O(N)
        """
        cur_node = head
        new_head = ListNode(head.val, head.next)
        new_cur_node = new_head
        list_len = 0

        while cur_node:
            list_len += 1
            cur_node = cur_node.next
            if cur_node:
                new_cur_node.next = ListNode(cur_node.val, cur_node.next)
            new_cur_node = new_cur_node.next

        rev_head = self.reverseList(new_head)
        for _ in range(math.floor(list_len/2)):
            if rev_head.val != head.val:
                return False
            rev_head = rev_head.next
            head = head.next

        return True

    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        """
        (without coying linked list, but this is kind of cheating)

        Time: O(N)
        Space: O(N)
        """
        all_vals = []

        cur_node = head
        while cur_node:
            all_vals.append(cur_node.val)
            cur_node = cur_node.next

        left, right = 0, len(all_vals)-1
        while left < right:
            if all_vals[left] != all_vals[right]:
                return False
            left += 1
            right -= 1

        return True

    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        """
        (in-place)

        Time: O(N)
        Space: O(1)
        """
        # fast and slow pointers to find the half of the list
        move_bit = 0
        prev_node = None
        fast_node = slow_node = head

        while fast_node:
            fast_node = fast_node.next
            if move_bit:
                prev_node = slow_node
                slow_node = slow_node.next
            move_bit = 1 if move_bit == 0 else 0

        # from head to prev_node is first half, slow_node to end is second half
        if not prev_node:
            return True

        # break first and second half and reverse second half
        prev_node.next = None
        rev_sec_half_head = self.reverseList(slow_node)

        # compare two halves
        while head:
            if head.val != rev_sec_half_head.val:
                return False
            head = head.next
            rev_sec_half_head = rev_sec_half_head.next

        return True

    def middleNode(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        Given the head of a singly linked list, return the middle node of the linked list.
        If there are two middle nodes, return the second middle node.

        Time: O(N)
        Space: O(1)
        """
        # fast and slow node as above
        move_bit = 0
        fast_node = slow_node = head

        while fast_node:
            fast_node = fast_node.next
            if move_bit:
                slow_node = slow_node.next
            move_bit = 1 if move_bit == 0 else 0

        return slow_node

    def hasCycle(self, head: Optional[ListNode]) -> bool:
        """
        Given head, the head of a linked list, determine if the linked list has a cycle in it.

        Time: O()
        Space: O()
        """
        # if fast and slow pointers meet, there is a cycle
        # fast and slow node as above
        move_bit = 0
        fast_node = slow_node = head

        while fast_node:
            fast_node = fast_node.next
            if move_bit:
                slow_node = slow_node.next
            move_bit = 1 if move_bit == 0 else 0

            if fast_node == slow_node:
                return True

        return False

    def reorderList(self, head: Optional[ListNode]) -> None:
        """
        Given linkedlist (head) as L0 → L1 → … → Ln - 1 → Ln, 
        modify it in-place to: L0 → Ln → L1 → Ln - 1 → L2 → Ln - 2 → …

        Time: O(N)
        Space: O(1)
        """
        # fast and slow pointers to find the half of the list
        move_bit = 0
        prev_node = None
        fast_node = slow_node = head

        while fast_node:
            fast_node = fast_node.next
            if move_bit:
                prev_node = slow_node
                slow_node = slow_node.next
            move_bit = 1 if move_bit == 0 else 0

        # from head to prev_node is first half, slow_node to end is second half
        if not prev_node:
            return

        # break first and second half and reverse second half
        prev_node.next = None
        rev_sec_head = self.reverseList(slow_node)

        # interweave two halves together
        cur_node = head
        cur_node_sec = rev_sec_head
        next_node = next_node_sec = prev_node_sec = None

        while cur_node:
            next_node = cur_node.next
            next_node_sec = cur_node_sec.next

            cur_node.next = cur_node_sec
            cur_node_sec.next = next_node

            cur_node = next_node
            prev_node_sec = cur_node_sec
            cur_node_sec = next_node_sec

        # second half might have one more node than first half
        if cur_node_sec:
            prev_node_sec.next = cur_node_sec

        return head

    def oddEvenList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        Given the head of a singly linked list, group all the nodes with odd indices together followed 
        by the nodes with even indices, and return the reordered list.
        e.x. head = [1,2,3,4,5] -> [1,3,5,2,4]

        Time: O(N)
        Space: O(1)
        """
        # if no node or only 1 node in linkedlist
        odd_head = head
        if not odd_head:
            return head
        even_head = head.next
        if not even_head:
            return head

        # move both odd and even pointers together
        cur_odd = odd_head
        cur_even = even_head
        next_odd = next_even = None
        while cur_even:
            if not cur_even.next:
                break

            next_even = cur_even.next.next
            next_odd = cur_odd.next.next

            cur_even.next = next_even
            cur_even = next_even

            cur_odd.next = next_odd
            cur_odd = next_odd

        # append even head next to odd tail
        cur_odd.next = even_head
        return head
