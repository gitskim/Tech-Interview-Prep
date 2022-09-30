from typing import List, Optional, Tuple


class TrieNode:

    def __init__(self, char='', is_end=False):
        self.val = char
        self.is_end = is_end
        self.children = [None] * 26  # the chars following this node

    def getChild(self, char: chr):
        """Check if it has a child TrieNode with char as val, return child node if exists."""
        return self.children[ord(char) - ord('a')]

    def addChild(self, char: chr, is_end=False):
        """Add (if needed) a child TrieNode with char to it's children."""
        index_of_char = ord(char) - ord('a')

        if not self.getChild(char):
            self.children[index_of_char] = TrieNode(char, is_end)
        if is_end:
            self.children[index_of_char].is_end = is_end

        return self.children[index_of_char]


class Trie:

    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        """
        Inserts the string word into the trie.

        Time: O(N) - N: len of word
        Space: O(N)
        """
        cur_node = self.root

        for c in word[:-1]:
            node_c = cur_node.addChild(c)
            cur_node = node_c
        cur_node.addChild(word[-1], is_end=True)

    def search(self, word: str) -> bool:
        """
        Returns true if the string word is in the trie (i.e., was inserted before), and false otherwise.

        Time: O(N) - N: len of word
        Space: O(1)
        """
        cur_node = self.root

        for c in word:
            node_c = cur_node.getChild(c)
            if not node_c:
                return False
            cur_node = node_c

        return cur_node and cur_node.is_end

    def startsWith(self, prefix: str) -> bool:
        """
        Returns true if there is a previously inserted string word that has the prefix prefix.

        Time: O(N) - N: len of word
        Space: O(1)
        """
        cur_node = self.root

        for c in prefix:
            node_c = cur_node.getChild(c)
            if not node_c:
                return False
            cur_node = node_c

        return True


class TrieNode:

    def __init__(self, char='', times=0):
        self.val = char
        self.times = times  # if times != 0, it's a leaf node

        self.children = {}  # all TrieNodes following this node, c->TrieNode

    def getChild(self, char: chr):
        """Check if it has a child TrieNode with char as val, return child node if exists."""
        if char in self.children:
            return self.children[char]
        return None

    def addChild(self, char: chr):
        """Add (if needed) a child TrieNode with char to it's children."""
        child_node = self.getChild(char)

        if not child_node:
            child_node = TrieNode(char)
            self.children[char] = child_node

        return child_node


class Trie:

    def __init__(self):
        self.root = TrieNode()

    def insert(self, sentence: str, times=1) -> None:
        """
        Inserts the string sentence into the trie.

        Time: O(N) - N: num of chars in sentence
        Space: O(N)
        """
        cur_node = self.root

        for c in sentence:
            node_c = cur_node.addChild(c)
            cur_node = node_c
        cur_node.times += times

    def startsWithTop3(self, prefix: str) -> List[str]:
        """
        Return at most 3 hottest sentences with prefix.

        Time: O(N + T)
        Space: O(T)
        """
        prefix_tail_node = self.startsWith(prefix)
        if not prefix_tail_node:
            return []

        all_sent = self.sentencesWithRootTop3(prefix_tail_node)
        if not all_sent:
            return []

        return [prefix[:-1] + sent[1] for sent in all_sent[:3]]

    def sentencesWithRootTop3(self, root: TrieNode) -> List[Tuple[str, int]]:
        """
        Return at most 3 hottest sentences and their occurance times that start with root node.

        Time: O(T) - num of TrieNodes in Trie
        Space: O(T)
        """
        # if itself if a leafnode
        all_sentences = []
        if root.times:
            all_sentences.append((-root.times, root.val))

        # if no children
        if not root.children:
            return all_sentences

        # append its val in front of all children's sentences
        for child in root.children.values():
            child_sentences = self.sentencesWithRootTop3(child)
            for (t, s) in child_sentences:
                all_sentences.append((t, root.val + s))

        all_sentences.sort()
        return all_sentences[:3]

    def startsWith(self, prefix: str) -> Optional[TrieNode]:
        """
        Returns TrieNode of last char in prefix if there is a previously inserted string word that has 
        the prefix prefix, None otherwise.

        Time: O(N) - N: len of word
        Space: O(1)
        """
        cur_node = self.root

        for c in prefix:
            node_c = cur_node.getChild(c)
            if not node_c:
                return None
            cur_node = node_c

        return cur_node


class AutocompleteSystem:

    def __init__(self, sentences: List[str], times: List[int]):
        """
        array of sentences and array of num of occurances of each sentence
        e.x. ["i love you", "island", "iroman", "i love leetcode"], [5, 3, 2, 2]

        Time: O(len_of_sentence * num_of_sentences)
        Space: O(len_of_sentence * num_of_sentences)
        """
        self.trie = Trie()
        for i in range(len(sentences)):
            self.trie.insert(sentences[i], times[i])

        self.input_sentence = ""

    def input(self, c: str) -> List[str]:
        """
        For each c input, return at most 3 sentences with prefix "..c" (ASCII order to break tie)
        When seeing "#", record this sentence in system with 1 time occurance.
        (previous calls of input() also recorded)

        e.x. input("i"), input(" "), input("a"), input("#")
             -> 3 hottest sentences with prefix: "i", "i ", "i a"
             -> and then at last record the sentence "i a" with occurance time 1

        Time: O(N + T)
        Space: O(T)
        """
        if c == "#":
            self.trie.insert(self.input_sentence, 1)
            self.input_sentence = ""
            return []

        self.input_sentence += c
        return self.trie.startsWithTop3(self.input_sentence)
