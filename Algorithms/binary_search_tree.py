class Node(object):
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

class BST(object):
    def __init__(self, root):
        self.root = Node(root)

    def insert(self, new_val):
        """ Ignoring repeated values? """
        node = self.root
        while node:
            # returns false if value already exists
            if node.value == new_val:
                return False
            # if value doesn't exist, keep traversing the tree
            elif node.value < new_val:
                if node.left:
                    node = node.left
                else:
                    node.left = Node(new_val)
            elif node.value > new_val:
                if node.right:
                    node = node.right
                else:
                    node.right = Node(new_val)

    def search(self, find_val):
        node = self.root
        while node:
            if node.value == find_val:
                return True
            elif node.value < find_val:
                node = node.left
            elif node.value > find_val:
                node = node.right

        return False

    def print_tree(self):
        """Print out all tree nodes
        as they are visited in
        a pre-order traversal."""
        values = self.preorder_print(self.root, [])
        return "-".join([str(i) for i in values])

    def preorder_print(self, node, traversal): # traversal?
        """Helper method - use this to create a 
        recursive print solution."""

        if node:
            if node.value:
                traversal = traversal + [node.value]
            if node.left: 
                traversal = self.preorder_print(node.left, traversal)
            if node.right: 
                traversal = self.preorder_print(node.right, traversal)
        return traversal    

# Set up tree
tree = BST(4)

# Insert elements
tree.insert(2)
tree.insert(1)
tree.insert(3)
tree.insert(5)

# Check search
# Should be True
print tree.search(4)
# Should be False
print tree.search(6)
print tree.print_tree()