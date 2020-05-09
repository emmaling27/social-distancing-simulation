import networkx as nx
import numpy as np


class Network():
    """ Generate a network using Toivonen et al.'s social network model """

    def __init__(self, n_0, n, mean_type, var_type):
        """
        n_0: initial seed of nodes
        n: number of nodes the network should have
        """
        assert(n_0 <= n)
        assert(n_0 > 0)
        self.n_0 = n_0
        self.n = n
        self.mean_type = mean_type
        self.var_type = var_type
        self.ic_rate = .016
        self.icf_rate = .08
        self.close_friends_rate = .1
        self.g = nx.Graph()
        self.rng = np.random.default_rng()

    def add_node(self, i):
        """ Adds node i to graph """
        self.g.add_node(i,
            senior=self.rng.binomial(1, .5),
            ic=self.rng.binomial(1, self.ic_rate),
            icf=self.rng.binomial(1, self.icf_rate))

    def add_edge(self, i, j):
        """ Adds edge (i, j) to graph """
        close = self.rng.binomial(1, self.close_friends_rate)
        value = self.rng.normal(self.mean_type[close], self.var_type[close])
        self.g.add_edge(i, j, close=close, value=value)

    def generate_network(self):
        """
        Generates the network using Toivonen et al.'s social network model
        """
        for node in range(self.n_0):
            self.add_node(node)
        for new_node in range(self.n_0, self.n):
            self.add_node(new_node)
            initial_contacts = self.rng.choice(
                np.array(self.g.nodes()),
                self.rng.binomial(1, .05) + 3,
                replace=False)
            secondary_pool = []
            for i in initial_contacts:
                secondary_pool += list(self.g.neighbors(i))
                self.add_edge(new_node, i)
            if secondary_pool:
                secondary_contacts = self.rng.choice(
                    secondary_pool,
                    min(self.rng.integers(1, 4), len(secondary_pool)),
                    replace=False)
                for i in secondary_contacts:
                    self.add_edge(new_node, i)
        return self.g

    def get_node_attrs(self, i):
        """ Returns a dictionary of senior, ic, icf attributes of node i """
        assert(i in self.g.nodes())
        node_attrs = {}
        for attr in ['senior', 'ic', 'icf']:
            node_attrs[attr] = bool(nx.get_node_attributes(self.g, attr)[i])
        return node_attrs

    def get_friendship_level(self, i, j):
        """ Returns true/false whether i and j are close friends """
        assert((i, j) in self.g.edges())
        return bool(
            nx.get_edge_attributes(self.g, 'close')[(min(i, j), max(i, j))])
