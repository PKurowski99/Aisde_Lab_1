from .node import Node
from .edge import Edge
from math import sqrt, inf
from matplotlib import pyplot as plt


def pythagoras(x, y):
    return int(f'{sqrt(x**2 + y**2):.0f}')


def create_adj_matrix(v, g):
    adj_matrix = []
    for i in range(0, v):
        adj_matrix.append([])
        for j in range(0, v):
            adj_matrix[i].append(0)

    # populate adjacency matrix with correct edge weights
    for i in range(0, len(g)):
        adj_matrix[g[i][0]][g[i][1]] = g[i][2]
        adj_matrix[g[i][1]][g[i][0]] = g[i][2]  # delete if edges are directed

    return adj_matrix


class Network(object):
    def __init__(self):
        pass

    algorithm = 'MST'
    nodes = []
    edges = []
    graph = {}
    nodes_to_compute = []

    def load_from_file(self):
        try:
            with open('input.txt', 'r') as input_file:
                for line in input_file:
                    if line.lstrip().startswith('#'):
                        continue
                    elif line.lstrip().startswith(('WEZLY', 'LACZA')):
                        try:
                            current = line.split('=')[0].strip(' \n')
                        except TypeError:
                            print('Matched string in WEZLY is not an int!')
                    elif line.lstrip().startswith('ALGORYTM'):
                        current = 'ALGORYTM'
                        self.algorithm = line.split('=')[-1].strip(' \n')
                    else:
                        if current == 'WEZLY':
                            params = [int(p.strip(' \n')) for p in line.split(' ')]
                            self.nodes.append(Node(*params))
                        elif current == 'LACZA':
                            params = [int(p.strip(' \n')) for p in line.split(' ')]
                            self.edges.append(Edge(*params))
                        else:
                            params = [int(p.strip(' \n')) for p in line.split(' ')]
                            self.nodes_to_compute.append((params[0], params[1]))
        except FileNotFoundError:
            print('No file called "input.txt" found!')

    def compute_weights(self):
        for x, edge in enumerate(self.edges):
            edge.weight = pythagoras(self.nodes[edge.start_node - 1].x - self.nodes[edge.end_node - 1].x,
                                     self.nodes[edge.start_node - 1].y - self.nodes[edge.end_node - 1].y)
            # print(f'{x+1:2}. {edge.weight:.2f}')  # f-string formatting

    def find_connection(self, start, goal):
        for edge in self.edges:
            if int(edge.start_node) == int(start) and int(edge.end_node) == int(goal):
                return edge.id_no
            # delete it when you need directed connections
            elif int(edge.end_node) == int(start) and int(edge.start_node) == int(goal):
                return edge.id_no
        return 0

    def draw(self, edges, nodes):
        for edge in edges:
            plt.plot([self.nodes[edge.start_node - 1].x, self.nodes[edge.end_node - 1].x],
                     [self.nodes[edge.start_node - 1].y, self.nodes[edge.end_node - 1].y],
                     'mediumorchid')
            plt.text((self.nodes[edge.start_node - 1].x + self.nodes[edge.end_node - 1].x) / 2,
                     (self.nodes[edge.start_node - 1].y + self.nodes[edge.end_node - 1].y) / 2,
                     edge.weight,  bbox=dict(boxstyle="round4,pad=0.3", fc='lightskyblue', alpha=0.5), fontsize=8)
        for node in nodes:
            plt.text(node.x, node.y, node.id_no, bbox=dict(boxstyle="circle,pad=0.3", fc="pink"), fontsize=12)
        plt.show()

    def draw_path(self, path):
        nodes = []
        edges = []
        for elements in path:
            nodes.append(self.nodes[int(elements) - 1])
        for i in range(nodes.__len__() - 1):
            edges.append(self.edges[self.find_connection(nodes[i].id_no, nodes[i + 1].id_no) - 1])
        self.draw(edges, nodes)

    def draw_graph_and_path(self, path):
        path_edges = []
        path_nodes = []
        for elements in path:
            path_nodes.append(self.nodes[int(elements) - 1])
        for i in range(path_nodes.__len__() - 1):
            path_edges.append(self.edges[self.find_connection(path_nodes[i].id_no, path_nodes[i + 1].id_no) - 1])
        for edge in self.edges:
            plt.plot([self.nodes[edge.start_node - 1].x, self.nodes[edge.end_node - 1].x],
                     [self.nodes[edge.start_node - 1].y, self.nodes[edge.end_node - 1].y],
                     'mediumorchid')
        for edge in path_edges:
            plt.plot([self.nodes[edge.start_node - 1].x, self.nodes[edge.end_node - 1].x],
                     [self.nodes[edge.start_node - 1].y, self.nodes[edge.end_node - 1].y],
                     'springgreen')
            plt.text((self.nodes[edge.start_node - 1].x + self.nodes[edge.end_node - 1].x) / 2,
                     (self.nodes[edge.start_node - 1].y + self.nodes[edge.end_node - 1].y) / 2,
                     edge.weight, bbox=dict(boxstyle="round4,pad=0.3",fc='lightskyblue', alpha=0.5), fontsize=10)
        for node in self.nodes:
            plt.text(node.x, node.y, node.id_no, bbox=dict(boxstyle="circle,pad=0.3", fc="pink"), fontsize=12)
        plt.show()

    def compute_graph(self):
        tmp_dict = {}
        for node in self.nodes:
            for edge in self.edges:
                if node.id_no == edge.start_node:
                    tmp_dict[str(edge.end_node)] = edge.weight
                elif node.id_no == edge.end_node:
                    tmp_dict[str(edge.start_node)] = edge.weight
            self.graph[str(node.id_no)] = tmp_dict.copy()
            tmp_dict.clear()

    def dijkstra(self, start, goal):
        self.compute_graph()
        shortest_distance = {}
        predecessor = {}
        unseen_nodes = self.graph
        infinity = inf
        path = []
        for node in unseen_nodes:
            shortest_distance[node] = infinity
        shortest_distance[start] = 0

        while unseen_nodes:
            min_node = None
            for node in unseen_nodes:
                if min_node is None:
                    min_node = node
                elif shortest_distance[node] < shortest_distance[min_node]:
                    min_node = node

            for edge, weight in self.graph[min_node].items():
                if weight + shortest_distance[min_node] < shortest_distance[edge]:
                    shortest_distance[edge] = weight + shortest_distance[min_node]
                    predecessor[edge] = min_node
            unseen_nodes.pop(min_node)

        current_node = goal
        while current_node != start:
            try:
                path.insert(0, current_node)
                current_node = predecessor[current_node]
            except KeyError:
                print('Path not reachable')
                break
        path.insert(0, start)
        if shortest_distance[goal] != infinity:
            print('Shortest distance is ' + str(shortest_distance[goal]))
            print('And the path is ' + str(path))
            self.draw_path(path)  # only path
            self.draw_graph_and_path(path)

    def prim(self):
        v = len(self.nodes)
        graph = []
        edges_mst = []
        total_weight = 0
        for edge in self.edges:
            graph.append([edge.start_node - 1, edge.end_node - 1, edge.weight])
        adj_matrix = create_adj_matrix(v, graph)
        vertex = 0
        mst = []
        edges = []
        visited = []
        min_edge = [None, None, float('inf')]
        while len(mst) != v - 1:

            # mark this vertex as visited
            visited.append(vertex)

            # add each edge to list of potential edges
            for r in range(0, v):
                if adj_matrix[vertex][r] != 0:
                    edges.append([vertex, r, adj_matrix[vertex][r]])

            # find edge with the smallest weight to a vertex
            # that has not yet been visited
            for e in range(0, len(edges)):
                if edges[e][2] < min_edge[2] and edges[e][1] not in visited:
                    min_edge = edges[e]

            # remove min weight edge from list of edges
            edges.remove(min_edge)

            # push min edge to MST
            mst.append(min_edge)

            # start at new vertex and reset min edge
            vertex = min_edge[1]
            min_edge = [None, None, float('inf')]
        for i in mst:
            j = self.find_connection(i[0] + 1, i[1] + 1) - 1
            edges_mst.append(self.edges[j])
            total_weight += self.edges[j].weight
        print(f"Total weight of the edges in minimum spanning tree: {total_weight}")
        self.draw(edges_mst, self.nodes)


