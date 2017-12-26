class Node(object):
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

class BinaryTree(object):
    def __init__(self, root):
        self.root = Node(root)

    def search(self, find_val):
        """Return True if the value
        is in the tree, return
        False otherwise."""
        return self.preorder_search(self.root, find_val)

    def print_tree(self):
        """Print out all tree nodes
        as they are visited in
        a pre-order traversal."""
        values = self.preorder_print(self.root, [])
        return "-".join([str(i) for i in values])

    def preorder_search(self, node, find_val): # changed start to node
        """Helper method - use this to create a 
        recursive search solution."""

        if node:
            if node.value == find_val:
                return True
            else:
                if node.left: 
                    found = self.preorder_search(node.left, find_val)
                    if found:
                        return True
                if node.right: 
                    return self.preorder_search(node.right, find_val)

        return False

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
tree = BinaryTree(1)
tree.root.left = Node(2)
tree.root.right = Node(3)
tree.root.left.left = Node(4)
tree.root.left.right = Node(5)

# Test search
#for i in range(1,11):
#    print tree.search(i)

# Test print_tree
# Should be 1-2-4-5-3
print tree.print_tree()