# question4.py

from time import time
from time import sleep
"""
Question 4
Find the least common ancestor between two nodes on a binary search tree. The least common ancestor is the farthest node from the root that is an ancestor of both nodes. For example, the root is a common ancestor of all nodes on the tree, but if both nodes are descendents of the root's left child, then that left child might be the lowest common ancestor. You can assume that both nodes are in the tree, and the tree itself adheres to all BST properties. The function definition should look like question4(T, r, n1, n2), where T is the tree represented as a matrix, where the index of the list is equal to the integer stored in that node and a 1 represents a child node, r is a non-negative integer representing the root, and n1 and n2 are non-negative integers representing the two nodes in no particular order. For example, one test case might be

question4([[0, 1, 0, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0],
           [1, 0, 0, 0, 1],
           [0, 0, 0, 0, 0]],
          3,
          1,lesa
          4)
and the answer would be 3.
"""

"""
Solution:
Find the path from root to each of the nodes. 
Then find the common ancestors between the two paths
Order the common ancestors ascending by their distance to the root, and return the first.
"""

def question4a(T, r, n1, n2):
    """ Find the least common ancestor between two nodes on a binary search tree """

    if len(T)<=2: return

    # custom function to find the path from root to n
    def path_to(search_node, root):
        """ Find the path from root to n"""
        
        node = root
        path = [node]
        while node != search_node:
            if search_node < node:
                # search the array from left to right to get the left node
                for i in range(len(T[node])):
                    if T[node][i] == 1:
                        path[:0] = [node]
                        node = i
                        break
            elif search_node > node:
                # search the array from right to left to get the right node
                for i in range(len(T[node])-1, -1, -1):
                    if T[node][i] == 1:
                        path[:0] = [node]
                        node = i
                        break
        return path

    # find path from root to n1 and n2
    path_to_n1 = path_to(n1, r)
    path_to_n2 = path_to(n2, r)

    #find the matches and its respectives distances from root
    common_ancestors = []
    for node1 in path_to_n1:
        for idx, node2 in enumerate(path_to_n2):
            if node1 == node2:
                common_ancestors.append((node2, idx))

    # find and return the common ancestor closest to root
    return sorted(common_ancestors, key=lambda x:x[1])[0][0]

"""
Reviewed Solution:
Traverse the tree from the rrot. If one of the nodes matches root, return root as least common ancestor. If not, explore the left and right subtree, recursively, until we find a node which has one key in the left subtree and another in the right subtree. 
"""

def question4b(T, r, n1, n2):
    """ Find the least common ancestor between two nodes on a binary search tree """

    # if three has less than three nodes, return None
    if len(T)<=2: return

    # recursive function to find least common ancestor
    def find_lca(r, n1, n2):

        # base case
        if r == None: 
            return None

        # if either n1 or n2 matches with root, returns root
        if r==n1 or r==n2:
            return r

        # search the array from left to right to get the left and right node
        left_node, right_node = None, None
        for i in range(len(T[r])):
            if T[r][i] == 1:
                if left_node == None:
                    left_node = i
                else:
                    right_node = i

        # look for keys in left and right subtrees
        left_lca = find_lca(left_node,n1,n2)
        right_lca = find_lca(right_node,n1,n2)

        # if both non-nul, the one key is present in each subtree
        if left_lca != None and right_lca != None:
            return r

        # otherwise check if left subtree or right subtree is lca
        if left_lca != None:
            return left_lca
        else:
            return right_lca

    return find_lca(r,n1,n2)


"""
Reviewed Solution:
Traverse the tree from the root. If one of the nodes matches root, return root as least common ancestor. 

If the left node (input node with the smallest value) is less than root, and right node (input node with highest value) is greater than root, than return root as least common ancestor.

If both left and right node are greather than root, set root to the right-most node descendant of root, and iterate. If both left and right node are smaller than root, set root to the left-most node descendant of root, and iterate.

This solution is based on two premises:
* It is a balanced tree
* Both n1 and n2 are in the tree
"""

def question4c(T, r, n1, n2):
    """ Find the least common ancestor between two nodes on a binary search tree """

    # if three has less than three nodes, return None
    if len(T)<=2: 
        return

    # order search nodes
    if n1<n2:
        ln, rn = n1, n2
    else:
        ln, rn = n2, n1

    while True:

        # if either node matches with root, return root
        if r==ln or r==ln:
            return r

        # if one node is in each side, return root
        if ln < r and rn > r:
            return r

        # if both on the right side, set root the right node and continue
        if ln > r and rn > r:
            # search the array from right to left to get the right node
            for i in range(len(T[r])-1, -1, -1):
                if T[r][i] == 1:
                    r = i
                    break

        # if both on the left side, set root the left node and continue
        if ln < r and rn < r:
            # search the array from left to right to get the left node
            for i in range(len(T[r])):
                if T[r][i] == 1:
                    r = i
                    break

versions = [question4a, question4b, question4c]

for question4 in versions:

    t0 = time()

    tree = [[0, 0, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 0, 0, 0]]
    print question4(tree, 3, 0, 2)
    # print 1

    tree = [[0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 0, 0]]
    print question4(tree, 2, 5, 3)
    # print 4

    tree = [[0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 1],
            [0, 0, 0, 0, 0]]
    print question4(tree, 3, 1, 4)
    # edge case: least common ancestor is root
    # print 3

    tree = [[0]]
    print question4(tree, 3, 1, 4)
    # edge case: empty tree or tree with less than three nodes
    # print None

    print "Version: {}, Time: {:.8f}".format(question4.__name__, time()-t0)


