# question3.py

"""
Given an undirected graph G, find the minimum spanning tree within G. A minimum spanning tree connects all vertices in a graph with the smallest possible total weight of edges. Your function should take in and return an adjacency list structured like this:

{'A': [('B', 2)],
 'B': [('A', 2), ('C', 5)], 
 'C': [('B', 5)]}
Vertices are represented as unique strings. The function definition should be question3(G)
"""

"""
Solution:  This solution is similar to a travel salesman problem, finding the shortest distance between two nodes. There are several optmized solutions for this problem, but the most simple one is to calculate the possible paths that connects all vertices and evaluating which one is shorter. A more optimized solution would be to use dynamic programming.

    The brute force solution is a graph traversal with all possible paths, which is implemented below. 
"""

def question3(G):
    """ Finds the minimum spanning tree within G """
    # I will consider it an acyclical graph for the simplest solution, we can remove this restriction later

    # create an array to store paths. Path will store visited nodes, traveled vertices and path cost
    paths = []

    # custom recursive function to traverse through all possible options
    def expand_path(current_node, visited_nodes, traveled_vertices, path_cost):
        vertices = G[current_node]
        for node, cost in vertices:
            if node not in visited_nodes:
                visited_nodes.append(node)
                traveled_vertices.append((node,cost))
                path_cost += cost
                # add to paths when finished
                if len(visited_nodes) == len(G):
                    paths.append((visited_nodes, traveled_vertices, path_cost))
                else:
                    expand_path(node, visited_nodes, traveled_vertices, path_cost)

    # construct a path starting from each node.
    for node, vertices in G.items():
        visited_nodes = [node]
        path_cost = 0
        traveled_vertices = []
        expand_path(node, visited_nodes, traveled_vertices, path_cost)

    # if no paths are found, return None
    if not paths: return

    # select shortest path
    shortest_path = sorted(paths, key= lambda x:x[1])[0]

    # convert to adjacency list format to output and return
    return { k:[v] for k,v in zip(shortest_path[0], shortest_path[1]) }


G = {'A': [('B', 7), ('C', 1)], 'B': [('A', 3), ('C', 2)], 'C': [('B', 5), ('A', 1)]}
print question3(G)
# a directed graph, with different vertex values depending on directionality 
# {'A': [('C', 1)], 'B': [('A', 3)]}

G = {}
print question3(G)
# edge case: empty graph returns no paths
# print None

G = {'A': [('B', 2)], 'B': [('A', 2), ('C', 5)], 'C': [('B', 5)]}
print question3(G)
# edge case: a non directed graph. all solutions have equal cost, return any of the possible combinations.
# print {'A': [('C', 5)], 'B': [('A', 2)]}
