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
        self.ic_rate = .016
        self.icf_rate = .08
        self.close_friends_rate = .1

    def generate_network(self):
        g = nx.Graph()
        rng = np.random.default_rng()
        g.add_nodes_from(range(self.n_0))
        for new_node in range(self.n_0, self.n):
            g.add_node(new_node,
                senior=rng.binomial(1, .5),
                ic=rng.binomial(1, self.ic_rate),
                icf=rng.binomial(1, self.icf_rate))
            c1 = rng.binomial(1, .05) + 1
            initial_contacts = rng.choice(
                np.array(g.nodes()),
                c1,
                replace=False)
            secondary_pool = []
            for i in initial_contacts:
                secondary_pool += list(g.neighbors(i))
                g.add_edge(new_node,
                    i,
                    close=rng.binomial(1, self.close_friends_rate))
            c2 = min(rng.integers(1, 4), len(secondary_pool))
            if secondary_pool:
                secondary_contacts = rng.choice(
                    secondary_pool,
                    c2,
                    replace=False)
                for i in secondary_contacts:
                    g.add_edge(new_node,
                        i,
                        close=rng.binomial(1, self.close_friends_rate))
        return g
