from math import inf


class Edge(object):
    def __init__(self, id_no, start_node, end_node, weight=inf):
        self.id_no = int(id_no)
        self.start_node = int(start_node)
        self.end_node = int(end_node)
        self.weight = weight
