from lab_1 import network


if __name__ == '__main__':
    n = network.Network()
    n.load_from_file()
    n.compute_weights()
    n.draw(n.edges, n.nodes)
    #n.dijkstra('1', '2')
    n.prim()
