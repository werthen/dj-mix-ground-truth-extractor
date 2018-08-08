"""General annotator module."""

from functools import reduce
from collections import defaultdict
import networkx as nx
import logging


class Annotator:
    def __init__(self, steps):
        self.G = nx.DiGraph()
        self.G.add_nodes_from(steps)

        need_dict = defaultdict(list)
        for s in steps:
            for need in s.needs:
                need_dict[need].append(s)

        for s in steps:
            for p in s.provides:
                for need_node in need_dict[p]:
                    self.G.add_edge(s, need_node)

                del need_dict[p]

        if len(need_dict) > 0:
            raise Exception('A need has not been met')

        if not nx.is_directed_acyclic_graph(self.G):
            raise Exception('The dependency graph contains a directed cycle')

    def start(self, **kwargs):
        empty_state = {}

        def process_node(state, node):
            logging.info(f"Starting {type(node).__name__}")
            return node.process(state)

        return reduce(process_node, nx.topological_sort(self.G), empty_state)
