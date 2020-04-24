import networkx as nx
import numpy as np


class Network():
    """
    Generate a network using Toivonen et al.'s social network model
    """

    def __init__(self, n_0, n, m_r=None, m_s=None):
        """
        n_0: initial seed of nodes
        n: number of nodes the network should have
        m_r: mean of initial contacts to connect a new node to
        m_s: mean of secondary contacts (friends of friends)
        """
        self.n_0 = n_0
        self.n = n
        self.m_r = m_r
        self.m_s = m_s

    def generate_network(self):
        g = nx.Graph()
        g.add_nodes_from(range(self.n_0))
        for new_node in range(self.n_0, self.n):
            c1 = np.random.binomial(1, .95) + 1
            rng = np.random.default_rng()
            initial_contacts = rng.choice(np.array(g.nodes()), c1, replace=False)
            secondary_pool = []
            for i in initial_contacts:
                secondary_pool += list(g.neighbors(i))
                g.add_edge(new_node, i)
            c2 = min(rng.integers(1, 4), len(secondary_pool))
            if secondary_pool:
                secondary_contacts = rng.choice(
                    secondary_pool,
                    c2,
                    replace=False)
                for i in secondary_contacts:
                    g.add_edge(new_node, i)
        return g
