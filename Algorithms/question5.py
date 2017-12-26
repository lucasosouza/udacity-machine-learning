# question5.py

"""
Question 5
Find the element in a singly linked list that's m elements from the end. For example, if a linked list has 5 elements, the 3rd element from the end is the 3rd element. The function definition should look like question5(ll, m), where ll is the first node of a linked list and m is the "mth number from the end". You should copy/paste the Node class below to use as a representation of a node in the linked list. Return the value of the node at that position.
"""

"""
Solution: A straighforward solution is to get the length of the linked list, and substract m from the length to the get element position in the list. After getting the position traverse the linked list again (length-m) times to get to the element.
"""

class Node(object):
    """ Node object (given) """
    
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList(object):
    """ Simple linked list implementation to test question5 """

    def __init__(self):
        self.root = None

    def append(self, new_node):
        """ Append a node to the end of the linked list """
        if not self.root:
            self.root = new_node
        else:
            node = self.root
            while node.next:
                node = node.next
            node.next = new_node

    def print_nodes(self):
        """ Print data for all nodes in linked list """
        output =  ''
        if self.root:
            node = self.root
            while node:
                output+=str(node.data)
                node = node.next
        print output

# create and test linked list
ll = LinkedList()
ll.append(Node(1))
ll.append(Node(2))
ll.append(Node(3))
ll.print_nodes()
# print 123

def question5(ll, m):
    """ Find the element in a singly linked list that's m elements from the end """

    # get linked list length
    if ll.root:
        node = ll.root
        ll_length = 1
        while node.next:
            ll_length+=1
            node = node.next
    else:
        return None

    # calculate node position in the list
    node_position = ll_length - m

    # if node position is negative or zero, return None
    if node_position <= 0:
        return None

    # traverse untill find position
    node = ll.root
    for _ in range(1, node_position):
        node = node.next
    return node.data

ll = LinkedList()
for i in range(11):
    ll.append(Node(i))
print question5(ll,3)
# print 7

ll = LinkedList()
print question5(ll,3)
# edge case: empty linked list. print None

ll = LinkedList()
ll.append(Node(1))
print question5(ll,1)
# edge case: only one element in list. print None

ll = LinkedList()
ll.append(Node(1))
ll.append(Node(2))
print question5(ll,3)
# edge case: not enough elements in list. print None




